#!/usr/bin/env python3
"""Compare a deployed RT-DETR ONNX export with its official PyTorch model."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from verify_predictions import difference, digest  # noqa: E402 - import after local src setup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-source",
        required=True,
        type=Path,
        help="pinned official rtdetr_pytorch directory",
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--reference-config", default="configs/rtdetr/rtdetr_r18vd_6x_coco.yml")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    for path in (
        args.checkpoint,
        args.config,
        args.model,
        args.image,
        args.reference_source / args.reference_config,
    ):
        if not path.is_file():
            parser.error(f"local file missing: {path}")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(args.report.parent / "huggingface-cache"))
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import cv2
    import numpy as np
    import onnxruntime as ort
    import torch
    import torchvision
    from PIL import Image
    from scipy.optimize import linear_sum_assignment
    from torchvision import transforms

    ort.set_default_logger_severity(3)
    from ai_inference import Config, Model, Prediction

    torch.set_num_threads(1)
    sys.path.insert(0, str(args.reference_source.resolve()))
    from src.core import YAMLConfig

    official_cfg = YAMLConfig(str(args.reference_source / args.reference_config))
    official_cfg.yaml_cfg["PResNet"]["pretrained"] = False
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
    official_cfg.model.load_state_dict(state, strict=True)
    reference_model = official_cfg.model.deploy().eval()
    reference_post = official_cfg.postprocessor.deploy()
    d = Config.from_yaml(args.config).model_dump()
    d["model"].update(path=args.model.resolve())
    d["runtime"]["options"] = {"providers": "cpu", "intra_op_num_threads": 1}
    # Exercise the official dynamic batch export, not just batch=1.
    d["runtime"]["batch"].update(mode="dynamic", size=2)
    config = Config.model_validate(d)
    from processing_reference import reference_settings

    pre_settings, post_settings = reference_settings(config, "rtdetr")
    model = Model(config)
    image = cv2.imread(str(args.image))
    if image is None:
        raise ValueError("cannot decode image")
    report: dict[str, Any] = dict(
        status="running",
        family="rtdetr",
        model_sha256=digest(args.model),
        checkpoint_sha256=digest(args.checkpoint),
        image_sha256=digest(args.image),
        reference_export_sha256=digest(args.reference_source / "tools/export_onnx.py"),
        reference_post_sha256=digest(
            args.reference_source / "src/zoo/rtdetr/rtdetr_postprocessor.py"
        ),
        torch=torch.__version__,
        torchvision=torchvision.__version__,
        onnxruntime=ort.__version__,
        opencv=cv2.__version__,
        providers=list(model.engine.providers),
        reference="unmodified official eager model + deployed postprocessor",
        scope="CPU real-weight implementation/export parity; not dataset accuracy or GPU acceptance",
        cases=[],
    )
    try:
        pre = pre_settings
        post = post_settings
        pipeline = transforms.Compose(
            [transforms.Resize((pre.height, pre.width)), transforms.ToTensor()]
        )
        variants = {
            "original": image,
            "flipped": cv2.flip(image, 1),
            "odd_size": cv2.resize(image, (813, 547)),
            "landscape": cv2.resize(image, (1001, 333)),
            "blank": np.zeros((481, 639, 3), np.uint8),
        }
        singles = []
        for name, bgr in variants.items():
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            source = rgb if pre.source_color == "RGB" else bgr
            expected_input = pipeline(Image.fromarray(rgb))[None]
            prepared = model.preprocess(source)
            np.testing.assert_array_equal(prepared.tensors[pre.input_name], expected_input.numpy())
            sizes = torch.tensor([[bgr.shape[1], bgr.shape[0]]], dtype=torch.int64)
            np.testing.assert_array_equal(prepared.tensors[pre.sizes_name], sizes.numpy())
            with torch.inference_mode():
                expected_raw = reference_post(reference_model(expected_input), sizes)
            labels, boxes, scores = [v.detach().cpu().numpy() for v in expected_raw]
            raw = model.infer(prepared)
            actual = model.postprocess(raw)[0]
            if not isinstance(actual, Prediction):
                raise TypeError("RT-DETR must return a detection result")
            singles.append(actual)
            # FP32 engine differences may reorder near-equal low-score top-k entries.
            # Verify all scores, then compare the selected paired detections below.
            np.testing.assert_allclose(raw.tensors[post.scores_name], scores, atol=2e-5, rtol=1e-4)
            actual_labels = raw.tensors[post.labels_name][0]
            actual_boxes = raw.tensors[post.boxes_name][0]
            actual_scores = raw.tensors[post.scores_name][0]
            scale = np.array([bgr.shape[1], bgr.shape[0]] * 2)
            cost = (
                np.abs(actual_boxes[:, None] / scale - boxes[0, None] / scale).mean(-1)
                + np.abs(actual_scores[:, None] - scores[0, None])
                + 2 * (actual_labels[:, None] != labels[0, None])
            )
            rows, cols = linear_sum_assignment(cost)
            np.testing.assert_array_equal(actual_labels[rows], labels[0, cols])
            np.testing.assert_allclose(actual_boxes[rows], boxes[0, cols], atol=0.05, rtol=1e-4)
            np.testing.assert_allclose(actual_scores[rows], scores[0, cols], atol=2e-5, rtol=1e-4)
            keep = scores[0] >= post.threshold
            expected_ids = labels[0, keep]
            expected_boxes = boxes[0, keep]
            if post.clip_boxes:
                expected_boxes = np.clip(expected_boxes, 0, [bgr.shape[1], bgr.shape[0]] * 2)
            expected_scores = scores[0, keep]
            np.testing.assert_array_equal(actual.class_ids, expected_ids)
            np.testing.assert_allclose(actual.boxes, expected_boxes, atol=0.05, rtol=1e-4)
            np.testing.assert_allclose(actual.scores, expected_scores, atol=2e-5, rtol=1e-4)
            report["cases"].append(
                dict(
                    name=name,
                    shape=list(bgr.shape),
                    detections=len(actual.boxes),
                    input_max_abs_error=0.0,
                    raw_scores_max_abs_error=difference(raw.tensors[post.scores_name], scores),
                    raw_boxes_max_abs_error=difference(actual_boxes[rows], boxes[0, cols]),
                    raw_classes_match=True,
                    boxes_max_abs_error=difference(actual.boxes, expected_boxes),
                    scores_max_abs_error=difference(actual.scores, expected_scores),
                    class_ids=actual.class_ids.tolist(),
                    boxes=actual.boxes.tolist(),
                    scores=actual.scores.tolist(),
                )
            )
            print(
                f"{name}: RT-DETR parity passed, {len(actual.boxes)} selected detections",
                flush=True,
            )
        sources = [
            cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if pre.source_color == "RGB" else bgr
            for bgr in variants.values()
        ]
        for i, result in enumerate(model.predict(sources)):
            if not isinstance(result, Prediction):
                raise TypeError("RT-DETR batch verification requires detection results")
            np.testing.assert_array_equal(result.class_ids, singles[i].class_ids)
            np.testing.assert_allclose(result.boxes, singles[i].boxes, atol=0.01, rtol=1e-4)
            np.testing.assert_allclose(result.scores, singles[i].scores, atol=2e-5, rtol=1e-4)
        assert model.predict([]) == []
        report.update(status="passed", dynamic_batch_counts=[2, 2, 1], empty_input_passed=True)
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
