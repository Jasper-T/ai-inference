#!/usr/bin/env python3
"""YOLO CPU real-weight parity checks; uses local files, never downloads models."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def difference(a, b):
    import numpy as np

    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))) if a.size else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    for path in (args.config, args.model, args.image):
        if not path.is_file():
            parser.error(f"local file missing: {path}")
    if args.model.suffix != ".onnx":
        parser.error("only existing local ONNX files are supported")
    import cv2
    import numpy as np
    import onnxruntime as ort

    ort.set_default_logger_severity(3)
    from ai_inference import Config, Model, Prediction

    image = cv2.imread(str(args.image))
    if image is None:
        raise ValueError("cannot decode input image")
    data = Config.from_yaml(args.config).model_dump()
    data["model"].update(path=args.model.resolve())
    data["runtime"]["options"] = {"providers": "cpu", "intra_op_num_threads": 1}
    config = Config.model_validate(data)
    from processing_reference import reference_settings

    pre_settings, post_settings = reference_settings(config, "yolo")
    model = Model(config)
    inputs, outputs = model.config.inputs, model.config.outputs
    if inputs is None or outputs is None:
        raise ValueError("model tensor contract must be resolved before verification")
    report = {
        "status": "running",
        "family": "yolo",
        "model_sha256": digest(args.model),
        "image_sha256": digest(args.image),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "onnxruntime": ort.__version__,
        "providers": list(model.engine.providers),
        "scope": "CPU real-weight implementation parity; not dataset accuracy or GPU acceptance",
        "cases": [],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    try:
        settings_dir = args.report.parent / "ultralytics-settings"
        settings_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("YOLO_CONFIG_DIR", str(settings_dir))
        os.environ.setdefault("YOLO_AUTOINSTALL", "false")
        from ultralytics import YOLO, __version__

        reference = YOLO(str(args.model.resolve()), task="detect")
        report["reference"] = f"Ultralytics {__version__} ONNX predictor"

        variants = {
            "original": image,
            "flipped": cv2.flip(image, 1),
            "odd_size": cv2.resize(image, (813, 547)),
            "landscape": cv2.resize(image, (1001, 333)),
            "blank": np.zeros((481, 639, 3), np.uint8),
        }
        for name, sample in variants.items():
            prepared = model.preprocess(sample)
            raw = model.infer(prepared)
            actual = model.postprocess(raw)[0]
            if not isinstance(actual, Prediction):
                raise TypeError("detection verification requires a detection result")
            post = post_settings
            expected = reference.predict(
                sample,
                imgsz=(pre_settings.height, pre_settings.width),
                rect=False,
                conf=post.confidence_threshold,
                iou=post.iou_threshold,
                max_det=post.max_detections,
                agnostic_nms=post.agnostic_nms,
                device="cpu",
                verbose=False,
                save=False,
                show=False,
            )[0]
            expected_tensor = reference.predictor.preprocess([sample]).cpu().numpy()
            np.testing.assert_array_equal(prepared.tensors[inputs[0].name], expected_tensor)
            expected_raw = reference.predictor.inference(reference.predictor.preprocess([sample]))
            if isinstance(expected_raw, (list, tuple)):
                expected_raw = expected_raw[0]
            if hasattr(expected_raw, "detach"):
                expected_raw = expected_raw.detach().cpu().numpy()
            actual_raw = raw.tensors[outputs[0].name]
            np.testing.assert_allclose(actual_raw, expected_raw, atol=1e-5, rtol=1e-4)
            ids = expected.boxes.cls.cpu().numpy().astype(np.int64)
            boxes = expected.boxes.xyxy.cpu().numpy()
            scores = expected.boxes.conf.cpu().numpy()
            np.testing.assert_array_equal(actual.class_ids, ids)
            np.testing.assert_allclose(actual.boxes, boxes, atol=0.05, rtol=1e-4)
            np.testing.assert_allclose(actual.scores, scores, atol=1e-5, rtol=1e-4)
            raw_error = difference(actual_raw, expected_raw)
            record = {
                "name": name,
                "shape": list(sample.shape),
                "detections": len(actual.boxes),
                "input_max_abs_error": 0.0,
                "raw_max_abs_error": raw_error,
                "boxes_max_abs_error": difference(actual.boxes, boxes),
                "scores_max_abs_error": difference(actual.scores, scores),
                "class_ids": actual.class_ids.tolist(),
                "boxes": actual.boxes.tolist(),
                "scores": actual.scores.tolist(),
            }
            report["cases"].append(record)
            print(f"{name}: parity passed, {len(actual.boxes)} detections", flush=True)
        samples = list(variants.values())
        single = [model.predict(s)[0] for s in samples]
        for count in (1, config.runtime.batch.size, config.runtime.batch.size + 1):
            request = [samples[i % len(samples)] for i in range(count)]
            for i, result in enumerate(model.predict(request)):
                expected = single[i % len(samples)]
                if not isinstance(result, Prediction) or not isinstance(expected, Prediction):
                    raise TypeError("batch verification requires detection results")
                np.testing.assert_array_equal(result.class_ids, expected.class_ids)
                np.testing.assert_allclose(result.boxes, expected.boxes, atol=1e-5, rtol=1e-5)
                np.testing.assert_allclose(result.scores, expected.scores, atol=1e-6, rtol=1e-5)
        assert model.predict([]) == []
        report.update(
            status="passed",
            batch_counts=[1, config.runtime.batch.size, config.runtime.batch.size + 1],
            empty_input_passed=True,
        )
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
