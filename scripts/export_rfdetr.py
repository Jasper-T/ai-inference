#!/usr/bin/env python3
"""Export the official RF-DETR Nano COCO checkpoint to ONNX."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("models/checkpoint_best_regular.pth")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    if hashlib.md5(checkpoint.read_bytes()).hexdigest() != "fb6504cce7fbdc783f7a46991f07639f":
        parser.error("checkpoint does not match the official Nano COCO MD5")

    from importlib.metadata import version

    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    from rfdetr import RFDETRNano

    torch.set_num_threads(1)
    model = RFDETRNano(pretrain_weights=str(checkpoint), device="cpu")
    sample = np.random.default_rng(0).normal(size=(1, 3, 384, 384)).astype(np.float32)
    inner = model.model.model.cpu().eval()
    with torch.inference_mode():
        reference = {
            key: value.cpu().numpy()
            for key, value in inner(torch.from_numpy(sample)).items()
            if isinstance(value, torch.Tensor)
        }
    output = model.export(
        output_dir=str(args.output_dir),
        output_name="rfdetr-nano",
        batch_size=1,
        dynamic_batch=False,
        verbose=False,
    )
    output = Path(output)
    onnx.checker.check_model(str(output))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(output), options, providers=["CPUExecutionProvider"])
    shape = session.get_inputs()[0].shape
    assert shape == list(sample.shape), shape
    actual = dict(
        zip([o.name for o in session.get_outputs()], session.run(None, {"input": sample}))
    )
    errors = {}
    for name, key in [("dets", "pred_boxes"), ("labels", "pred_logits")]:
        expected = reference[key]
        tensor = actual[name]
        if not isinstance(tensor, np.ndarray):
            raise TypeError("RF-DETR ONNX outputs must be ndarrays")
        np.testing.assert_allclose(tensor, expected, atol=2e-4, rtol=2e-4)
        errors[name] = float(np.abs(tensor - expected).max())
    record = {
        "source": "https://repo.roboflow.com/rfdetr/nano_coco/checkpoint_best_regular.pth",
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "onnx_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "rfdetr_version": version("rfdetr"),
        "torch_version": torch.__version__,
        "inputs": {o.name: o.shape for o in session.get_inputs()},
        "outputs": {o.name: o.shape for o in session.get_outputs()},
        "max_absolute_error": errors,
        "status": "graph_checked_and_raw_output_parity_passed",
    }
    output.with_suffix(".export.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
