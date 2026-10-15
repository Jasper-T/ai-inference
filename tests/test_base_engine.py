import numpy as np
import pytest

from ai_inference import BaseEngine, Config, OMEngine, ONNXRuntimeEngine


class EchoEngine(BaseEngine):
    calls = 0

    def _infer(self, tensors):
        self.calls += 1
        return {"y": tensors["x"]}


def make_config(config, dynamic=False):
    data = config.model_dump()
    if dynamic:
        data["runtime"]["batch"]["mode"] = "dynamic"
    n = None if dynamic else 4
    data["inputs"] = [dict(name="x", dtype="float32", shape=(n, 2))]
    data["outputs"] = [dict(name="y", dtype="float32", shape=(n, 2))]
    return Config.model_validate(data)


@pytest.mark.parametrize("dynamic,count", [(False, 4), (True, 1), (True, 4)])
def test_valid_batch(config, dynamic, count):
    engine = EchoEngine(make_config(config, dynamic))
    x = np.ones((count, 2), np.float32)
    np.testing.assert_array_equal(engine.infer({"x": x})["y"], x)
    assert engine.calls == 1


@pytest.mark.parametrize(
    "inputs",
    [
        {},
        {"x": 1},
        {"x": np.array(1)},
        {"x": np.zeros((0, 2), np.float32)},
        {"x": np.zeros((3, 2), np.float32)},
        {"x": np.zeros((5, 2), np.float32)},
        {"wrong": np.zeros((4, 2), np.float32)},
        {"x": np.zeros((4, 2), np.float16)},
        {"x": np.zeros((4, 3), np.float32)},
    ],
)
def test_bad_input_never_calls_backend(config, inputs):
    engine = EchoEngine(make_config(config))
    with pytest.raises(ValueError):
        engine.infer(inputs)
    assert engine.calls == 0


@pytest.mark.parametrize(
    "outputs", [None, {}, {"y": np.zeros((3, 2), np.float32)}, {"y": np.zeros((4, 2), np.float16)}]
)
def test_output_contract(config, outputs):
    engine = EchoEngine(make_config(config))
    engine._infer = lambda tensors: outputs
    with pytest.raises(ValueError):
        engine.infer({"x": np.zeros((4, 2), np.float32)})


def test_error_passthrough_and_inheritance(config):
    error = RuntimeError("backend error")
    engine = EchoEngine(make_config(config))

    def fail(tensors):
        raise error

    engine._infer = fail
    with pytest.raises(RuntimeError) as caught:
        engine.infer({"x": np.zeros((4, 2), np.float32)})
    assert caught.value is error
    assert ONNXRuntimeEngine.infer is BaseEngine.infer
    assert OMEngine.infer is BaseEngine.infer


@pytest.mark.parametrize("dynamic,batch", [(False, 4), (True, 1)])
def test_direct_engine_warmup_and_metadata(config, dynamic, batch):
    engine = EchoEngine(make_config(config, dynamic))
    seen = []

    def infer(tensors):
        seen.append(tensors["x"].copy())
        return {"y": tensors["x"]}

    engine._infer = infer
    engine.warmup()
    assert len(seen) == engine.config.runtime.warmup_runs
    assert all(x.shape == (batch, 2) and x.dtype == np.float32 and not x.any() for x in seen)
    metadata = engine.read_tensor_metadata()
    assert metadata["inputs"][0].name == "x"
    assert metadata["outputs"][0].shape == (None if dynamic else 4, 2)
    assert ONNXRuntimeEngine.warmup is BaseEngine.warmup
    assert OMEngine.warmup is BaseEngine.warmup
