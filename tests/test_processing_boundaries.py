"""Processing placement and public model boundaries without request statistics."""

import numpy as np
import pytest

from ai_inference import BaseEngine, Config, EngineRegistry, Model, Pipeline


class ProcessingEngine(BaseEngine):
    def __init__(self, config):
        self.config = config
        self.calls = 0

    def _infer(self, tensors):
        self.calls += 1
        n = len(tensors["input"])
        return {
            "dets": np.zeros((n, 300, 4), np.float32),
            "labels": np.zeros((n, 300, 2), np.float32),
        }


EngineRegistry.register("processing_test", ProcessingEngine)


@pytest.fixture
def model(config):
    data = config.model_dump()
    data["runtime"].update(backend="processing_test", warmup=False)
    return Model(Config.model_validate(data))


@pytest.mark.parametrize("stage,device", [("preprocess", "cuda"), ("postprocess", "trt")])
def test_unimplemented_placement_rejected_before_engine(config, stage, device, monkeypatch):
    data = config.model_dump()
    data[stage]["device"] = device

    def unexpected(*args):
        pytest.fail("must reject placement before allocating a session")

    monkeypatch.setattr(EngineRegistry, "create", unexpected)
    with pytest.raises(ValueError, match="not implemented"):
        Model(Config.model_validate(data))


def test_cpu_placement_independent_of_predict_provider(model):
    assert model.preprocessor.settings.device == "cpu"
    assert model.postprocessor.settings.device == "cpu"
    assert model.predictor.engine is model.engine
    assert isinstance(model.preprocessor, Pipeline)
    assert isinstance(model.postprocessor, Pipeline)


def test_model_uses_pipeline_directly(model):
    assert isinstance(model.preprocessor, Pipeline)
    assert isinstance(model.postprocessor, Pipeline)
    assert not hasattr(model, "adapter")
