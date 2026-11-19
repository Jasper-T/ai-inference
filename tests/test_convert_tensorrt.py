"""Validate conversion profiles and builder failures without NVIDIA libraries."""

import argparse
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    "convert_tensorrt", Path(__file__).resolve().parents[1] / "scripts/convert_tensorrt.py"
)
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)


@pytest.mark.parametrize(
    "value", ["", "x=0x2:4x2:8x2", "x=4x2:2x2:8x2", "x=2:4x2:8", "x=2:4:8;x=2:4:8"]
)
def test_invalid_profile(value):
    with pytest.raises(argparse.ArgumentTypeError):
        converter.parse_profile(value)


def test_multi_input_profiles_and_fixed_dimension_checks():
    x = SimpleNamespace(name="x", shape=(-1, 2), is_shape_tensor=False)
    aux = SimpleNamespace(name="aux", shape=(-1, 3), is_shape_tensor=False)
    network = SimpleNamespace(num_inputs=2, get_input=lambda i: (x, aux)[i])
    seen = []
    builder = SimpleNamespace(
        create_optimization_profile=lambda: SimpleNamespace(
            set_shape=lambda *args: seen.append(args)
        )
    )
    config = SimpleNamespace(add_optimization_profile=lambda p: 0)
    profile = converter.parse_profile("x=2x2:4x2:8x2;aux=2x3:4x3:8x3")
    converter.configure_profiles(builder, network, config, [profile])
    assert seen == [("x", (2, 2), (4, 2), (8, 2)), ("aux", (2, 3), (4, 3), (8, 3))]
    for profiles in (
        [],
        [{"x": profile["x"]}],
        [{**profile, "unknown": profile["x"]}],
        [{**profile, "aux": ((2, 4), (4, 4), (8, 4))}],
    ):
        with pytest.raises(ValueError):
            converter.configure_profiles(builder, network, config, profiles)


def test_builder_returns_serialized_bytes_or_error(tmp_path):
    args = SimpleNamespace(onnx=tmp_path / "x.onnx", workspace_mib=64, precision="fp16", profile=[])
    settings, flags = [], []
    config = SimpleNamespace(
        set_memory_pool_limit=lambda *args: settings.append(args),
        clear_flag=lambda flag: None,
        set_flag=flags.append,
    )
    network = SimpleNamespace(
        num_inputs=1,
        get_input=lambda _: SimpleNamespace(name="x", shape=(2, 2), is_shape_tensor=False),
    )
    builder = SimpleNamespace(
        create_network=lambda flags: network,
        create_builder_config=lambda: config,
        build_serialized_network=lambda *args: b"engine",
    )
    parser = SimpleNamespace(parse_from_file=lambda path: True)
    trt = SimpleNamespace(
        Logger=type("Logger", (), {"WARNING": 1, "__init__": lambda *args: None}),
        init_libnvinfer_plugins=lambda *args: True,
        Builder=lambda logger: builder,
        OnnxParser=lambda *args: parser,
        MemoryPoolType=SimpleNamespace(WORKSPACE=1),
        BuilderFlag=SimpleNamespace(FP16=2, TF32=3),
    )
    assert converter.build(args, trt) == b"engine"
    assert settings == [(1, 64 * 1024**2)] and flags == [2]
    builder.build_serialized_network = lambda *args: None
    with pytest.raises(RuntimeError, match="build failed"):
        converter.build(args, trt)
    parser.parse_from_file = lambda path: False
    parser.num_errors = 1
    parser.get_error = lambda i: "unsupported ONNX node"
    with pytest.raises(RuntimeError, match="unsupported ONNX node"):
        converter.build(args, trt)
