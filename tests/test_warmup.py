import numpy as np
import pytest

from ai_inference import BaseEngine, Config, Model
from ai_inference.model import EngineRegistry


@pytest.mark.parametrize("enabled,runs,expected", [(None, None, 3), (False, 5, 0), (True, 2, 2)])
def test_warmup_counts(config, monkeypatch, enabled, runs, expected):
    data = config.model_dump()
    data["runtime"].pop("warmup")
    data["runtime"].pop("warmup_runs")
    if enabled is not None:
        data["runtime"]["warmup"] = enabled
    if runs is not None:
        data["runtime"]["warmup_runs"] = runs
    calls = []

    class Engine(BaseEngine):
        def _infer(self, tensors):
            calls.append(tensors["input"].copy())
            n = tensors["input"].shape[0]
            return {
                "dets": np.zeros((n, 300, 4), np.float32),
                "labels": np.zeros((n, 300, 2), np.float32),
            }

    monkeypatch.setattr(EngineRegistry, "create", lambda backend, config: Engine(config))
    Model(Config.model_validate(data))
    assert len(calls) == expected
    for x in calls:
        assert x.shape == (4, 3, 560, 560) and x.dtype == np.float32 and not x.any()


def test_dynamic_multi_input_and_error(config, monkeypatch):
    data = config.model_dump()
    data["runtime"].update(warmup=True, warmup_runs=1)
    data["runtime"]["batch"]["mode"] = "dynamic"
    data["inputs"] = [
        dict(name="a", dtype="float16", shape=(None, 2)),
        dict(name="b", dtype="int64", shape=(None, 3)),
    ]
    data["outputs"] = [dict(name="out", dtype="float32", shape=(None, 2))]
    calls = []

    class Engine(BaseEngine):
        def _infer(self, tensors):
            calls.append(tensors)
            return {"out": np.zeros((tensors["a"].shape[0], 2), np.float32)}

    engine = Engine(Config.model_validate(data))
    engine.warmup()
    assert calls[0]["a"].shape == (1, 2) and calls[0]["a"].dtype == np.float16
    assert calls[0]["b"].shape == (1, 3) and calls[0]["b"].dtype == np.int64
    data["inputs"][0]["shape"] = (None, None)
    engine.config = Config.model_validate(data)
    with pytest.raises(ValueError, match="non-batch"):
        engine.warmup()
    assert len(calls) == 1


def test_warmup_error_propagates(config, monkeypatch):
    data = config.model_dump()
    data["runtime"]["warmup"] = True
    error = RuntimeError("inference failure")

    class Engine(BaseEngine):
        def _infer(self, tensors):
            raise error

    monkeypatch.setattr(EngineRegistry, "create", lambda backend, config: Engine(config))
    with pytest.raises(RuntimeError) as caught:
        Model(Config.model_validate(data))
    assert caught.value is error


@pytest.mark.parametrize("runs", [0, -1, True, 1.5])
def test_invalid_warmup_runs(config, runs):
    data = config.model_dump()
    data["runtime"]["warmup_runs"] = runs
    with pytest.raises(ValueError):
        Config.model_validate(data)
