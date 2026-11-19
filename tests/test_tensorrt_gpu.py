"""Opt-in real GPU conversion and execution; skips on ordinary developer machines."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from ai_inference import Config, DefaultPredictor, PreparedInput, TensorRTEngine

if os.environ.get("AI_INFERENCE_TENSORRT_GPU_TESTS") != "1":
    pytest.skip(
        "set AI_INFERENCE_TENSORRT_GPU_TESTS=1 on an NVIDIA GPU host", allow_module_level=True
    )

onnx = pytest.importorskip("onnx")
pytest.importorskip("tensorrt")
pytest.importorskip("cuda.bindings")


@pytest.mark.parametrize("dynamic", [False, True])
def test_real_tensorrt_conversion_execution_and_profile_switch(tmp_path, dynamic):
    shape = ["batch", "width"] if dynamic else [2, 2]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Add", ["x", "aux"], ["y"])],
        "addition",
        [
            onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape)
            for name in ("x", "aux")
        ],
        [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, shape)],
    )
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 17)])
    model.ir_version = 9
    source, target = tmp_path / "add.onnx", tmp_path / "add.engine"
    onnx.save(model, source)
    command = [
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "scripts/convert_tensorrt.py"),
        str(source),
        str(target),
    ]
    if dynamic:
        command += [
            "--profile",
            "x=2x2:4x2:8x2;aux=2x2:4x2:8x2",
            "--profile",
            "x=2x4:4x4:8x4;aux=2x4:4x4:8x4",
        ]
    subprocess.run(command, check=True, timeout=300)
    config = Config(
        model={"name": "add", "path": target},
        runtime={"backend": "tensorrt", "warmup": False},
        preprocess={"steps": []},
        postprocess={"steps": []},
    )
    engine = TensorRTEngine(config)
    try:
        assert engine.config.runtime.batch.size == (4 if dynamic else 2)
        for count, width in [(5, 2), (3, 4), (1, 2)] if dynamic else [(3, 2)]:
            x = np.arange(count * width, dtype=np.float32).reshape(count, width)
            aux = x + 3
            prepared = PreparedInput({"x": x, "aux": aux}, tuple(range(count)))
            raw = DefaultPredictor(engine, engine.config)(prepared)
            np.testing.assert_allclose(raw.tensors["y"], x + aux, atol=1e-6, rtol=1e-6)
    finally:
        engine.close()
    engine.close()
