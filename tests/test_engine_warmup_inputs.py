import numpy as np
import pytest
from test_batcher_predictor import contract
from test_tensor_boundaries import RecorderEngine

from ai_inference import Config, Model
from ai_inference.engines.base import BaseEngine


class RecordingEcho(BaseEngine):
    def __init__(self, config):
        super().__init__(config)
        self.inputs = []

    def _infer(self, tensors):
        self.inputs.append({name: tensor.copy() for name, tensor in tensors.items()})
        return {"y": tensors["x"]}


def test_supplied_warmup_resolves_unknown_dimensions(config):
    data = contract(config, dynamic=True).model_dump()
    for kind in ("inputs", "outputs"):
        data[kind][0]["shape"] = (None, None)
    engine = RecordingEcho(Config.model_validate(data))
    with pytest.raises(ValueError, match="non-batch"):
        engine.warmup()
    assert engine.inputs == []
    values = np.ones((2, 7), np.float32)
    engine.warmup({"x": values})
    assert len(engine.inputs) == engine.config.runtime.warmup_runs
    for tensors in engine.inputs:
        np.testing.assert_array_equal(tensors["x"], values)


@pytest.mark.parametrize(
    "values",
    [np.ones((1, 2), np.float32), np.ones((4, 2), np.float16), np.ones((4, 3), np.float32)],
)
def test_invalid_supplied_warmup_never_executes_backend(config, values):
    engine = RecordingEcho(contract(config))
    with pytest.raises(ValueError):
        engine.warmup({"x": values})
    assert engine.inputs == []


def test_engine_can_create_semantically_valid_multi_input_warmup(config):
    data = contract(config).model_dump()
    data["inputs"] = [
        {"name": "x", "dtype": "float32", "shape": (4, 2)},
        {"name": "indices", "dtype": "int64", "shape": (4, 1)},
    ]

    class IndexEngine(RecordingEcho):
        def create_warmup_inputs(self):
            return {"x": np.ones((4, 2), np.float32), "indices": np.ones((4, 1), np.int64)}

        def _infer(self, tensors):
            if not (tensors["indices"] > 0).all():
                raise ValueError("positive indices required")
            return super()._infer(tensors)

    engine = IndexEngine(Config.model_validate(data))
    engine.warmup()
    assert len(engine.inputs) == engine.config.runtime.warmup_runs
    assert all(
        tensors["indices"].dtype == np.int64 and tensors["indices"].min() == 1
        for tensors in engine.inputs
    )


def test_model_delegates_warmup_and_accepts_supplied_tensors(config, monkeypatch):
    class OwnWarmupEngine(RecorderEngine):
        def __init__(self, config):
            super().__init__(config)
            self.warmups = []

        def create_warmup_inputs(self):
            pytest.fail("custom warmup must own its input preparation")

        def warmup(self, tensors=None):
            self.warmups.append(tensors)

    monkeypatch.setattr(
        "ai_inference.model.EngineRegistry.create", lambda backend, config: OwnWarmupEngine(config)
    )
    data = config.model_dump()
    data["runtime"]["warmup"] = True
    model = Model(Config.model_validate(data))
    assert model.engine.warmups == [None]
    tensors = {"input": np.ones((4, 3, 560, 560), np.float32)}
    model.warmup(tensors)
    assert model.engine.warmups[1] is tensors
    assert model.predictor.batcher is model.batcher
