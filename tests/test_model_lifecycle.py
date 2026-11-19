"""Model ownership, including engines that predate explicit close support."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event
from types import SimpleNamespace

import numpy as np
import pytest

from ai_inference import BaseEngine, EngineRegistry, Model


def test_model_context_closes_engine_once_and_rejects_operations(config, monkeypatch):
    class OwnedEngine(BaseEngine):
        closed = 0

        def close(self):
            self.closed += 1

    engine = OwnedEngine(config)
    monkeypatch.setattr(EngineRegistry, "create", lambda *args: engine)
    with Model(config) as model:
        assert model.engine is engine
    model.close()
    assert engine.closed == 1
    with pytest.raises(RuntimeError, match="closed"):
        model.predict(np.zeros((10, 10, 3), np.uint8))
    with pytest.raises(RuntimeError, match="closed"):
        model.warmup()


def test_model_warmup_failure_closes_engine_preserving_original_error(config, monkeypatch):
    class BrokenEngine(BaseEngine):
        closed = 0

        def warmup(self):
            raise ValueError("warmup failed")

        def close(self):
            self.closed += 1
            raise RuntimeError("cleanup failed")

    config = config.model_copy(
        update={"runtime": config.runtime.model_copy(update={"warmup": True})}
    )
    engine = BrokenEngine(config)
    monkeypatch.setattr(EngineRegistry, "create", lambda *args: engine)
    with pytest.raises(ValueError, match="warmup failed"):
        Model(config)
    assert engine.closed == 1


def test_legacy_engine_without_close_is_still_accepted(config, monkeypatch):
    monkeypatch.setattr(EngineRegistry, "create", lambda *args: SimpleNamespace(config=config))
    model = Model(config)
    model.close()
    model.close()


def test_parallel_operations_are_preserved_and_close_waits(config, monkeypatch):
    entered = Barrier(3)
    release, closing = Event(), Event()

    class OwnedEngine(BaseEngine):
        closed = False

        def close(self):
            self.closed = True

    engine = OwnedEngine(config)
    monkeypatch.setattr(EngineRegistry, "create", lambda *args: engine)
    model = Model(config)
    original = model.preprocessor

    def blocked(images):
        entered.wait(timeout=5)
        assert release.wait(5)
        return original(images)

    model.preprocessor = blocked

    def close():
        closing.set()
        model.close()

    with ThreadPoolExecutor(max_workers=3) as pool:
        runs = [pool.submit(model.preprocess, np.zeros((10, 10, 3), np.uint8)) for _ in range(2)]
        entered.wait(timeout=5)
        done = pool.submit(close)
        assert closing.wait(5)
        assert not done.done() and not engine.closed
        release.set()
        for run in runs:
            assert run.result(timeout=5).count == 1
        done.result(timeout=5)
    assert engine.closed
