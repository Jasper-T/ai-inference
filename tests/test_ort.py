import sys
from types import SimpleNamespace

import numpy as np
import pytest

from ai_inference import Config, ONNXRuntimeEngine


def fake_ort(monkeypatch, tmp_path, config, available=None, active=None, mismatch=False):
    path = tmp_path / "model.onnx"
    path.touch()
    data = config.model_dump()
    data["model"]["path"] = path
    config = Config.model_validate(data)
    state = {}

    class Options:
        def add_session_config_entry(self, k, v):
            state[k] = v

    class Session:
        def __init__(self, path, sess_options, providers):
            state["options"] = sess_options
            state["providers"] = providers

        def disable_fallback(self):
            state["disabled"] = True

        def get_providers(self):
            return active if active is not None else ["CUDAExecutionProvider"]

        def get_inputs(self):
            assert config.inputs is not None
            return [
                SimpleNamespace(
                    name="bad" if mismatch else s.name, shape=s.shape, type="tensor(float)"
                )
                for s in config.inputs
            ]

        def get_outputs(self):
            assert config.outputs is not None
            return [
                SimpleNamespace(name=s.name, shape=s.shape, type="tensor(float)")
                for s in config.outputs
            ]

    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(
            SessionOptions=Options,
            InferenceSession=Session,
            get_available_providers=lambda: (
                available if available is not None else ["CUDAExecutionProvider"]
            ),
        ),
    )
    return config, state


def test_strict_cuda_options(monkeypatch, tmp_path, config):
    config, state = fake_ort(monkeypatch, tmp_path, config)
    engine = ONNXRuntimeEngine(config)
    assert state["session.disable_cpu_ep_fallback"] == "1"
    assert state["disabled"] is True
    assert state["providers"] == [("CUDAExecutionProvider", {"device_id": "0"})]
    assert engine.providers == ("CUDAExecutionProvider",)


def test_unavailable_cuda_rejected(monkeypatch, tmp_path, config):
    config, _ = fake_ort(monkeypatch, tmp_path, config, available=["CPUExecutionProvider"])
    with pytest.raises(RuntimeError, match="unavailable"):
        ONNXRuntimeEngine(config)


def test_cuda_initialization_cpu_fallback_rejected(monkeypatch, tmp_path, config):
    config, _ = fake_ort(monkeypatch, tmp_path, config, active=["CPUExecutionProvider"])
    with pytest.raises(RuntimeError, match="initialization failed"):
        ONNXRuntimeEngine(config)


def test_cuda_registration_with_cpu_graph_fallback_disabled(monkeypatch, tmp_path, config):
    config, state = fake_ort(
        monkeypatch, tmp_path, config, active=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    ONNXRuntimeEngine(config)
    assert state["session.disable_cpu_ep_fallback"] == "1"


def test_io_metadata_checked_at_initialization(monkeypatch, tmp_path, config):
    config, _ = fake_ort(monkeypatch, tmp_path, config, mismatch=True)
    with pytest.raises(ValueError, match="names"):
        ONNXRuntimeEngine(config)


def test_explicit_mixed_graph(monkeypatch, tmp_path, config):
    data = config.model_dump()
    data["runtime"]["options"]["providers"] = [
        {"name": "TensorrtExecutionProvider", "options": {"trt_fp16_enable": "1"}},
        {"name": "CUDAExecutionProvider", "options": {"device_id": "0"}},
        {"name": "CPUExecutionProvider"},
    ]
    config, state = fake_ort(
        monkeypatch,
        tmp_path,
        Config.model_validate(data),
        available=["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
        active=["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    ONNXRuntimeEngine(config)
    assert "session.disable_cpu_ep_fallback" not in state
    assert state["providers"] == [
        ("TensorrtExecutionProvider", {"trt_fp16_enable": "1"}),
        ("CUDAExecutionProvider", {"device_id": "0"}),
        ("CPUExecutionProvider", {}),
    ]
    assert state["disabled"]


def test_real_ort_cpu_identity(tmp_path, config):
    import onnx
    from onnx import TensorProto, helper

    path = tmp_path / "identity.onnx"
    graph = helper.make_graph(
        [helper.make_node("Identity", ["x"], ["y"])],
        "identity",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [4, 2])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [4, 2])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, path)
    data = config.model_dump()
    data["model"]["path"] = path
    data["runtime"]["options"].update(providers=[{"name": "CPUExecutionProvider"}])
    data["inputs"] = [dict(name="x", dtype="float32", shape=(4, 2))]
    data["outputs"] = [dict(name="y", dtype="float32", shape=(4, 2))]
    engine = ONNXRuntimeEngine(Config.model_validate(data))
    x = np.arange(8, dtype=np.float32).reshape(4, 2)
    np.testing.assert_array_equal(engine.infer({"x": x})["y"], x)
    with pytest.raises(ValueError):
        engine.infer({"x": x[:1]})


@pytest.mark.parametrize(
    "field,value",
    [
        ("provider", "CPUExecutionProvider"),
        ("intra_op_num_threads", 2),
        ("inter_op_num_threads", 1),
        ("provider_options", {}),
        ("allow_cpu_fallback", True),
    ],
)
def test_reject_legacy_runtime_fields(config, field, value):
    data = config.model_dump()
    data["runtime"][field] = value
    with pytest.raises(ValueError):
        Config.model_validate(data)


@pytest.mark.parametrize(
    "options",
    [
        {"typo": 1},
        {"provider": "CPUExecutionProvider"},
        {"intra_op_num_threads": -1},
        {"inter_op_num_threads": 0},
        {"inter_op_num_threads": 1},
        {"inter_op_num_threads": 4},
    ],
)
def test_engine_validates_options_before_loading(config, options):
    data = config.model_dump()
    data["runtime"]["options"] = options
    cfg = Config.model_validate(data)
    with pytest.raises(ValueError):
        ONNXRuntimeEngine(cfg)


def test_custom_backend_options_are_not_ort_options(config):
    data = config.model_dump()
    data["runtime"].update(backend="custom", options={"custom_setting": 123})
    assert Config.model_validate(data).runtime.options == {"custom_setting": 123}


@pytest.mark.parametrize(
    "providers",
    [
        [],
        [{"name": ""}],
        [{"name": "CPUExecutionProvider"}, {"name": "CPUExecutionProvider"}],
        [{"name": "CPUExecutionProvider", "typo": 1}],
    ],
)
def test_invalid_provider_config(config, providers):
    data = config.model_dump()
    data["runtime"]["options"] = {"providers": providers}
    with pytest.raises(ValueError):
        ONNXRuntimeEngine(Config.model_validate(data))


def test_ort_initialization_error_propagates(monkeypatch, tmp_path, config):
    config, _ = fake_ort(monkeypatch, tmp_path, config)
    error = RuntimeError("ORT failed")

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(sys.modules["onnxruntime"], "InferenceSession", fail)
    with pytest.raises(RuntimeError) as caught:
        ONNXRuntimeEngine(config)
    assert caught.value is error


@pytest.mark.parametrize(
    "shape,dtype", [([4, 2], "tensor(float)"), ([4, 3, 560, 560], "tensor(int64)")]
)
def test_ort_metadata_shape_and_dtype(monkeypatch, tmp_path, config, shape, dtype):
    config, _ = fake_ort(monkeypatch, tmp_path, config)
    session = sys.modules["onnxruntime"].InferenceSession
    monkeypatch.setattr(
        session, "get_inputs", lambda self: [SimpleNamespace(name="input", shape=shape, type=dtype)]
    )
    with pytest.raises(ValueError):
        ONNXRuntimeEngine(config)


@pytest.mark.parametrize(
    "value,names",
    [
        ("cpu", ["CPUExecutionProvider"]),
        ("cuda", ["CUDAExecutionProvider"]),
        ("trt", ["TensorrtExecutionProvider"]),
        (
            ["trt", "cuda", "cpu"],
            ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
        ),
    ],
)
def test_short_provider_syntax(monkeypatch, tmp_path, config, value, names):
    d = config.model_dump()
    d["runtime"]["options"]["providers"] = value
    c, state = fake_ort(
        monkeypatch, tmp_path, Config.model_validate(d), available=names, active=names
    )
    engine = ONNXRuntimeEngine(c)
    assert [name for name, _ in state["providers"]] == names
    assert engine.requested_providers == tuple(names)
    assert engine.selected_providers == tuple(names)


@pytest.mark.parametrize(
    "value", ["", "gpu", [], ["cuda", "cuda"], ["cuda", "CUDAExecutionProvider"]]
)
def test_invalid_short_providers(config, value):
    d = config.model_dump()
    d["runtime"]["options"]["providers"] = value
    with pytest.raises(ValueError):
        ONNXRuntimeEngine(Config.model_validate(d))


def test_explicit_chain_skips_unavailable_cuda(monkeypatch, tmp_path, config):
    d = config.model_dump()
    d["runtime"]["options"]["providers"] = ["cuda", "cpu"]
    c, state = fake_ort(
        monkeypatch,
        tmp_path,
        Config.model_validate(d),
        available=["CPUExecutionProvider"],
        active=["CPUExecutionProvider"],
    )
    engine = ONNXRuntimeEngine(c)
    assert state["providers"] == [("CPUExecutionProvider", {})]
    assert engine.requested_providers == ("CUDAExecutionProvider", "CPUExecutionProvider")
    assert engine.selected_providers == ("CPUExecutionProvider",)


def test_chain_initialization_failure_tries_next(monkeypatch, tmp_path, config):
    d = config.model_dump()
    d["runtime"]["options"]["providers"] = ["cuda", "cpu"]
    c, state = fake_ort(
        monkeypatch,
        tmp_path,
        Config.model_validate(d),
        available=["CUDAExecutionProvider", "CPUExecutionProvider"],
        active=["CPUExecutionProvider"],
    )
    cls = sys.modules["onnxruntime"].InferenceSession
    original = cls.__init__
    attempts = []

    def init(self, path, sess_options, providers):
        attempts.append([n for n, _ in providers])
        if providers[0][0] == "CUDAExecutionProvider":
            raise RuntimeError("CUDA library failed")
        original(self, path, sess_options, providers)

    monkeypatch.setattr(cls, "__init__", init)
    engine = ONNXRuntimeEngine(c)
    assert attempts == [["CUDAExecutionProvider", "CPUExecutionProvider"], ["CPUExecutionProvider"]]
    assert engine.selected_providers == ("CPUExecutionProvider",)


def test_chain_rejects_implicit_primary_fallback_then_uses_explicit_cpu(
    monkeypatch, tmp_path, config
):
    d = config.model_dump()
    d["runtime"]["options"]["providers"] = ["cuda", "cpu"]
    c, state = fake_ort(
        monkeypatch,
        tmp_path,
        Config.model_validate(d),
        available=["CUDAExecutionProvider", "CPUExecutionProvider"],
        active=["CPUExecutionProvider"],
    )
    cls = sys.modules["onnxruntime"].InferenceSession
    original = cls.__init__
    attempts = []

    def init(self, path, sess_options, providers):
        attempts.append([n for n, _ in providers])
        original(self, path, sess_options, providers)

    monkeypatch.setattr(cls, "__init__", init)
    engine = ONNXRuntimeEngine(c)
    assert attempts == [["CUDAExecutionProvider", "CPUExecutionProvider"], ["CPUExecutionProvider"]]
    assert engine.selected_providers == ("CPUExecutionProvider",)


def test_mixed_chain_contract_error_does_not_retry(monkeypatch, tmp_path, config):
    d = config.model_dump()
    d["runtime"]["options"]["providers"] = ["cuda", "cpu"]
    c, state = fake_ort(
        monkeypatch,
        tmp_path,
        Config.model_validate(d),
        available=["CUDAExecutionProvider", "CPUExecutionProvider"],
        active=["CUDAExecutionProvider", "CPUExecutionProvider"],
        mismatch=True,
    )
    cls = sys.modules["onnxruntime"].InferenceSession
    original = cls.__init__
    attempts = []

    def init(self, *args, **kwargs):
        attempts.append(1)
        original(self, *args, **kwargs)

    monkeypatch.setattr(cls, "__init__", init)
    with pytest.raises(ValueError, match="names"):
        ONNXRuntimeEngine(c)
    assert len(attempts) == 1


def test_execution_error_does_not_retry_cpu(monkeypatch, tmp_path, config):
    d = config.model_dump()
    d["runtime"]["options"]["providers"] = ["cuda", "cpu"]
    c, state = fake_ort(
        monkeypatch,
        tmp_path,
        Config.model_validate(d),
        available=["CUDAExecutionProvider", "CPUExecutionProvider"],
        active=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    calls = []
    error = RuntimeError("GPU execution failed")

    def run(self, *args, **kwargs):
        calls.append(1)
        raise error

    monkeypatch.setattr(sys.modules["onnxruntime"].InferenceSession, "run", run, raising=False)
    engine = ONNXRuntimeEngine(c)
    assert c.inputs is not None
    tensors = {}
    for spec in c.inputs:
        assert spec.shape is not None
        tensors[spec.name] = np.zeros(tuple(d or 1 for d in spec.shape), dtype=spec.dtype)
    with pytest.raises(RuntimeError) as caught:
        engine.infer(tensors)
    assert caught.value is error and len(calls) == 1
