"""Backend-independent ownership and rollback, including direct Engine callers."""

import weakref
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event

import numpy as np
import pytest
from test_ort import fake_ort

from ai_inference import BaseEngine, OMEngine, ONNXRuntimeEngine, TensorRTEngine


def input_tensors(config):
    return {spec.name: np.zeros(spec.shape, spec.dtype) for spec in config.inputs}


def output_tensors(config):
    return {spec.name: np.zeros(spec.shape, spec.dtype) for spec in config.outputs}


@pytest.mark.parametrize("engine_type", [ONNXRuntimeEngine, OMEngine, TensorRTEngine])
def test_builtin_backends_share_public_lifecycle(engine_type):
    assert engine_type.close is BaseEngine.close
    assert engine_type.infer is BaseEngine.infer
    assert engine_type.warmup is BaseEngine.warmup
    assert engine_type._close is not BaseEngine._close


def test_base_default_cleanup_and_closed_guard(config):
    engine = BaseEngine(config)
    engine.close()
    engine.close()
    # Closed state takes precedence over bad tensors and warmup preparation.
    with pytest.raises(RuntimeError, match="closed"):
        engine.infer({})
    with pytest.raises(RuntimeError, match="closed"):
        engine.warmup()


def test_legacy_subclass_that_sets_config_without_base_init(config):
    class LegacyEngine(BaseEngine):
        def __init__(self, config):
            self.config = config

        def _infer(self, tensors):
            return output_tensors(self.config)

    engine = LegacyEngine(config)
    assert engine.infer(input_tensors(config))
    engine.close()
    with pytest.raises(RuntimeError, match="closed"):
        engine.infer(input_tensors(config))


@pytest.mark.parametrize("fails", [False, True])
def test_concurrent_close_runs_cleanup_once_even_on_error(config, fails):
    started, release = Event(), Event()

    class Engine(BaseEngine):
        calls = 0

        def _close(self):
            self.calls += 1
            started.set()
            assert release.wait(5)
            if fails:
                raise RuntimeError("cleanup failure")

    engine = Engine(config)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(engine.close)
        assert started.wait(5)
        second = pool.submit(engine.close)
        release.set()
        if fails:
            with pytest.raises(RuntimeError, match="cleanup failure"):
                first.result(timeout=5)
        else:
            first.result(timeout=5)
        second.result(timeout=5)
    assert engine.calls == 1
    engine.close()
    with pytest.raises(RuntimeError, match="closed"):
        engine.infer({})


def test_close_preserves_parallel_infer_and_waits_for_all_calls(config):
    entered = Barrier(3)
    release, closing = Event(), Event()

    class Engine(BaseEngine):
        closed = False

        def _infer(self, tensors):
            entered.wait(timeout=5)
            assert release.wait(5)
            assert not self.closed
            return output_tensors(self.config)

        def _close(self):
            self.closed = True

    engine = Engine(config)

    def close():
        closing.set()
        engine.close()

    with ThreadPoolExecutor(max_workers=3) as pool:
        runs = [pool.submit(engine.infer, input_tensors(config)) for _ in range(2)]
        entered.wait(timeout=5)
        done = pool.submit(close)
        assert closing.wait(5)
        # Synchronize on the admission state, not a wall-clock sleep.
        with engine._lifecycle_condition:
            engine._lifecycle_condition.wait_for(lambda: engine._closing, timeout=5)
            assert engine._closing and not engine._closed
        with pytest.raises(RuntimeError, match="closed"):
            engine.infer(input_tensors(config))
        assert not done.done()
        release.set()
        for run in runs:
            run.result(timeout=5)
        done.result(timeout=5)
    assert engine.closed


def test_close_waits_for_entire_warmup_and_allows_nested_infer(config):
    entered, release, closing = Event(), Event(), Event()

    class Engine(BaseEngine):
        calls = 0

        def _infer(self, tensors):
            self.calls += 1
            if self.calls == 1:
                entered.set()
                assert release.wait(5)
            return output_tensors(self.config)

    engine = Engine(config)

    def close():
        closing.set()
        engine.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        warming = pool.submit(engine.warmup)
        assert entered.wait(5)
        done = pool.submit(close)
        assert closing.wait(5)
        with engine._lifecycle_condition:
            engine._lifecycle_condition.wait_for(lambda: engine._closing, timeout=5)
            assert engine._closing
        release.set()
        warming.result(timeout=5)
        done.result(timeout=5)
    assert engine.calls == config.runtime.warmup_runs


def test_reentrant_close_is_rejected_instead_of_deadlocking(config):
    class Engine(BaseEngine):
        def _infer(self, tensors):
            self.close()

    engine = Engine(config)
    with pytest.raises(RuntimeError, match="active engine operation"):
        engine.infer(input_tensors(config))
    engine.close()


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_initialization_failure_preserves_original_and_releases_resource(config, cleanup_fails):
    closed = []

    class Engine(BaseEngine):
        def __init__(self, config):
            super().__init__(config)
            with self._initialization():
                raise ValueError("initialization failure")

        def _close(self):
            closed.append(self)
            if cleanup_fails:
                raise RuntimeError("secondary cleanup error")

    with pytest.raises(ValueError, match="initialization failure"):
        Engine(config)
    assert len(closed) == 1
    closed[0].close()


@pytest.mark.parametrize("broken", [False, True])
def test_ort_releases_owned_session_on_close_or_metadata_failure(config, monkeypatch, broken):
    sessions = []

    def load(self, runtime):
        self.session = object()
        sessions.append(self)

    monkeypatch.setattr(ONNXRuntimeEngine, "_load_session", load)

    def contract(self):
        if broken:
            raise ValueError("invalid metadata")

    monkeypatch.setattr(ONNXRuntimeEngine, "_load_tensor_contract", contract)
    if broken:
        with pytest.raises(ValueError, match="invalid metadata"):
            ONNXRuntimeEngine(config)
        assert sessions[0]._closed
    else:
        engine = ONNXRuntimeEngine(config)
        assert engine.session is not None
        engine.close()
    assert sessions[0].session is None


def test_ort_rejected_provider_session_is_not_kept_by_traceback(config, monkeypatch, tmp_path):
    import sys

    config, _ = fake_ort(monkeypatch, tmp_path, config, active=["CPUExecutionProvider"])
    session_type = sys.modules["onnxruntime"].InferenceSession
    original = session_type.__init__
    references = []

    def initialize(self, *args, **kwargs):
        references.append(weakref.ref(self))
        original(self, *args, **kwargs)

    monkeypatch.setattr(session_type, "__init__", initialize)
    with pytest.raises(RuntimeError, match="initialization failed") as caught:
        ONNXRuntimeEngine(config)
    assert caught.traceback is not None  # Keep the failure traceback alive deliberately.
    assert len(references) == 1 and references[0]() is None


@pytest.mark.parametrize("broken,cleanup_fails", [(False, False), (True, False), (True, True)])
def test_om_calls_instance_release_once_and_never_global_finalize(
    config, monkeypatch, broken, cleanup_fails
):
    calls, engines = [], []

    class Session:
        def free_resource(self):
            calls.append("free_resource")
            if cleanup_fails:
                raise RuntimeError("release failure")

        def finalize(self):
            pytest.fail("must not finalize the process-wide ACL runtime")

    def load(self, options):
        self.session = Session()
        engines.append(self)

    def contract(self):
        if broken:
            raise ValueError("invalid metadata")

    monkeypatch.setattr(OMEngine, "_load_session", load)
    monkeypatch.setattr(OMEngine, "_load_tensor_contract", contract)
    config = config.model_copy(
        update={"runtime": config.runtime.model_copy(update={"backend": "om", "options": {}})}
    )
    if broken:
        with pytest.raises(ValueError, match="invalid metadata"):
            OMEngine(config)
    else:
        OMEngine(config).close()
    engines[0].close()
    assert calls == ["free_resource"] and engines[0].session is None
