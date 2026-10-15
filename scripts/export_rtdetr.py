#!/usr/bin/env python3
"""Export a local official RT-DETR checkpoint into the SDK deploy contract."""

import argparse
import inspect
import json
import os
import sys
import types
from pathlib import Path

from verify_predictions import digest


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
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    for p in (args.checkpoint, args.reference_source / args.reference_config):
        if not p.is_file():
            parser.error(f"local file missing: {p}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(args.output.parent / "huggingface-cache"))
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import onnx
    import torch

    torch.set_num_threads(1)
    sys.path.insert(0, str(args.reference_source.resolve()))
    from src.core import YAMLConfig

    cfg = YAMLConfig(str(args.reference_source / args.reference_config))
    cfg.yaml_cfg["PResNet"]["pretrained"] = False
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
    cfg.model.load_state_dict(state, strict=True)

    class Deployed(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.post = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            return self.post(self.model(images), orig_target_sizes)

    model = Deployed().eval()
    fallback = False
    if torch.__version__.startswith("2.0."):
        # Torch2.0 has no ONNX symbolic for optimized SDPA at opset16.
        # Only export uses explicit attention; verify_rtdetr compares the ONNX
        # output against the unmodified official eager path afterwards.
        for module in model.modules():
            if isinstance(module, torch.nn.MultiheadAttention):
                original = module.forward

                def explicit(self, *args, _original=original, **kwargs):
                    kwargs["need_weights"] = True
                    return _original(*args, **kwargs)

                module.forward = types.MethodType(explicit, module)
                fallback = True
    options = dict(
        input_names=["images", "orig_target_sizes"],
        output_names=["labels", "boxes", "scores"],
        dynamic_axes={
            name: {0: "N"} for name in ("images", "orig_target_sizes", "labels", "boxes", "scores")
        },
        opset_version=16,
    )
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        options["dynamo"] = False
    with torch.inference_mode():
        torch.onnx.export(
            model,
            (torch.zeros(1, 3, 640, 640), torch.tensor([[640, 640]], dtype=torch.int64)),
            str(args.output),
            **options,
        )
    onnx.checker.check_model(onnx.load(args.output))
    record = dict(
        checkpoint_sha256=digest(args.checkpoint),
        model_sha256=digest(args.output),
        export_source_sha256=digest(args.reference_source / "tools/export_onnx.py"),
        torch=torch.__version__,
        opset=16,
        static_size=[640, 640],
        dynamic_batch=True,
        explicit_attention_export_fallback=fallback,
        status="exported_and_graph_checked; requires verify_rtdetr parity check",
    )
    args.output.with_suffix(".export.json").write_text(json.dumps(record, indent=2))
    print(f"Exported: {args.output}\nSHA256: {record['model_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
