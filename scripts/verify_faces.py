#!/usr/bin/env python3
"""Compare local YuNet/SFace ONNX files with OpenCV's CPU reference APIs."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from verify_predictions import difference, digest  # noqa: E402 - import after local src setup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True, choices=("yunet", "sface"))
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument(
        "--landmarks", type=Path, help="SFace: JSON [[x,y],...] in original YuNet five-point order"
    )
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="mark reports for generated probe graphs, not trained weights",
    )
    args = parser.parse_args()
    for p in (args.config, args.model, args.image):
        if not p.is_file():
            parser.error(f"local file missing: {p}")
    if args.family == "sface" and (args.landmarks is None or not args.landmarks.is_file()):
        parser.error("SFace requires --landmarks")
    import cv2
    import numpy as np
    import onnxruntime as ort

    ort.set_default_logger_severity(3)
    from ai_inference import Config, FaceInput, FeaturePrediction, Model, Prediction
    from ai_inference.processing.decoders.sface import align_face

    image = cv2.imread(str(args.image))
    if image is None:
        raise ValueError("cannot decode image")
    d = Config.from_yaml(args.config).model_dump()
    d["model"].update(path=args.model.resolve())
    d["runtime"]["options"] = {"providers": "cpu"}
    config = Config.model_validate(d)
    from processing_reference import reference_settings

    pre_settings, post_settings = reference_settings(config, args.family)
    model = Model(config)
    outputs = model.config.outputs
    if outputs is None:
        raise ValueError("model output tensor contract must be resolved before verification")
    report: dict[str, Any] = dict(
        status="running",
        family=args.family,
        model_sha256=digest(args.model),
        image_sha256=digest(args.image),
        opencv=cv2.__version__,
        onnxruntime=ort.__version__,
        providers=list(model.engine.providers),
        reference="OpenCV CPU DNN / FaceDetectorYN / FaceRecognizerSF",
        scope=(
            "synthetic OpenCV implementation probe; no trained-weight verification"
            if args.synthetic
            else "local-weight implementation parity; not recognition accuracy or GPU acceptance"
        ),
        cases=[],
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    try:
        pre = pre_settings
        post = post_settings
        if args.family == "yunet":
            detector = cv2.FaceDetectorYN.create(
                str(args.model),
                "",
                (pre.width, pre.height),
                post.threshold,
                post.nms_threshold,
                post.top_k,
                cv2.dnn.DNN_BACKEND_OPENCV,
                cv2.dnn.DNN_TARGET_CPU,
            )
            net = cv2.dnn.readNetFromONNX(str(args.model))
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            samples = {
                "original": image,
                "odd_size": cv2.resize(image, (813, 547)),
                "flipped": cv2.flip(image, 1),
                "blank": np.zeros((481, 639, 3), np.uint8),
            }
            for name, bgr in samples.items():
                source = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if pre.source_color == "RGB" else bgr
                prepared = model.preprocess(source)
                raw = model.infer(prepared)
                actual = model.postprocess(raw)[0]
                if not isinstance(actual, Prediction):
                    raise TypeError("YuNet must return a detection result")
                resized = cv2.resize(bgr, (pre.width, pre.height))
                reference_tensor = cv2.dnn.blobFromImage(resized)
                np.testing.assert_array_equal(prepared.tensors[pre.input_name], reference_tensor)
                names = [s.name for s in outputs]
                net.setInput(reference_tensor)
                reference_raw = dict(zip(names, net.forward(names)))
                raw_errors = {}
                for key, values in raw.tensors.items():
                    expected = np.asarray(reference_raw[key]).reshape(values.shape)
                    np.testing.assert_allclose(values, expected, atol=1e-4, rtol=1e-3)
                    raw_errors[key] = difference(values, expected)
                _, faces = detector.detect(resized)
                faces = (
                    np.empty((0, 15), np.float32) if faces is None else faces[: post.max_detections]
                )
                scale = np.array([bgr.shape[1] / pre.width, bgr.shape[0] / pre.height], np.float32)
                boxes = np.concatenate(
                    (faces[:, :2], faces[:, :2] + faces[:, 2:4]), axis=1
                ) * np.tile(scale, 2)
                if post.clip_boxes:
                    boxes = np.clip(boxes, 0, [bgr.shape[1], bgr.shape[0]] * 2)
                points = faces[:, 4:14].reshape(-1, 5, 2) * scale
                if actual.keypoints is None:
                    raise ValueError("YuNet verification requires face keypoints")
                np.testing.assert_allclose(actual.boxes, boxes, atol=0.1, rtol=1e-3)
                np.testing.assert_allclose(actual.keypoints, points, atol=0.1, rtol=1e-3)
                np.testing.assert_allclose(actual.scores, faces[:, 14], atol=1e-4, rtol=1e-3)
                report["cases"].append(
                    dict(
                        name=name,
                        detections=len(actual.boxes),
                        input_max_abs_error=0.0,
                        raw_errors=raw_errors,
                        boxes_max_abs_error=difference(actual.boxes, boxes),
                        keypoints_max_abs_error=difference(actual.keypoints, points),
                    )
                )
                print(f"{name}: YuNet parity passed", flush=True)
        else:
            points = np.asarray(json.loads(args.landmarks.read_text()), np.float32)
            if points.shape != (5, 2) or not np.isfinite(points).all():
                raise ValueError("five finite xy landmarks required")
            recognizer = cv2.FaceRecognizerSF.create(
                str(args.model), "", cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU
            )
            source = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if pre.source_color == "RGB" else image
            face = np.concatenate((np.zeros(4), points.ravel(), [1])).astype(np.float32)[None]
            expected_crop = recognizer.alignCrop(image, face)
            actual_crop = align_face(image, points)
            delta = np.abs(actual_crop.astype(np.float64) - expected_crop.astype(np.float64))
            # NumPy/OpenCV SVD rounding can change a small number of interpolated pixels.
            if delta.mean() > 0.02 or delta.max() > 8:
                raise AssertionError("alignment differs beyond documented pixel tolerances")
            prepared = model.preprocess(FaceInput(source, points))
            reference_tensor = cv2.dnn.blobFromImage(
                actual_crop, 1, (112, 112), (0, 0, 0), swapRB=True, crop=False
            )
            np.testing.assert_array_equal(prepared.tensors[pre.input_name], reference_tensor)
            raw = model.infer(prepared)
            result = model.postprocess(raw)[0]
            if not isinstance(result, FeaturePrediction):
                raise TypeError("SFace must return a feature result")
            actual = result.embedding.astype(np.float64)
            # Compare engines on the identical SDK crop, separate from alignment drift.
            ref_same_crop = recognizer.feature(actual_crop).reshape(1, -1)
            np.testing.assert_allclose(
                raw.tensors[post.output_name], ref_same_crop, atol=1e-3, rtol=1e-3
            )
            reference_feature = recognizer.feature(expected_crop).reshape(-1).astype(np.float64)
            expected = reference_feature / np.linalg.norm(reference_feature)
            normalized = actual / np.linalg.norm(actual)
            cosine = float(normalized @ expected)
            if cosine < 0.9999:
                raise AssertionError(f"SFace cosine parity below .9999: {cosine}")
            report["cases"].append(
                dict(
                    name="aligned_face",
                    input_max_abs_error=0.0,
                    max_pixel_error=float(delta.max()),
                    mean_pixel_error=float(delta.mean()),
                    raw_max_abs_error=difference(raw.tensors[post.output_name], ref_same_crop),
                    cosine=cosine,
                    normalized_feature_max_abs_error=difference(normalized, expected),
                )
            )
            print(f"SFace parity passed; cosine={cosine:.8f}", flush=True)
        report["status"] = "passed"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
