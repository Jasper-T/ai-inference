from dataclasses import fields

import numpy as np
import pytest
from scripts.benchmark import benchmark

from ai_inference import PreparedInput, RawOutput


class FakeModel:
    def __init__(self):
        self.calls = []

    def predict(self, images):
        self.calls.append("warmup")
        return []

    def preprocess(self, images):
        self.calls.append("preprocess")
        return PreparedInput({"x": np.zeros((1, 2), np.float32)}, (object(),))

    def infer(self, prepared):
        self.calls.append("predict")
        return RawOutput({"y": prepared.tensors["x"]}, prepared)

    def postprocess(self, raw):
        self.calls.append("postprocess")
        return []


def test_benchmark_excludes_warmup_and_measures_public_stages(monkeypatch):
    ticks = iter([1.0, 1.1, 1.3, 1.6, 2.0, 2.2, 2.6, 3.0])
    monkeypatch.setattr("scripts.benchmark.perf_counter", lambda: next(ticks))
    model = FakeModel()
    syncs = []
    report = benchmark(
        model, [object(), object()], runs=2, warmup=2, synchronize=lambda: syncs.append(1)
    )
    assert model.calls == ["warmup"] * 2 + ["preprocess", "predict", "postprocess"] * 2
    assert len(syncs) == 10
    assert report["latency_unit"] == "ms"
    assert report["latency"]["preprocess"]["mean"] == pytest.approx(150)
    assert report["latency"]["total"]["p50"] == pytest.approx(800)
    assert report["latency"]["total"]["p95"] == pytest.approx(980)
    assert report["throughput_samples_per_second"] == pytest.approx(2.5)


@pytest.mark.parametrize("kwargs", [{"runs": 0}, {"runs": True}, {"warmup": -1}, {"warmup": 1.5}])
def test_benchmark_invalid_options_never_execute_model(kwargs):
    model = FakeModel()
    with pytest.raises(ValueError):
        benchmark(model, object(), **kwargs)
    assert not model.calls


def test_benchmark_propagates_failure_without_success_report():
    model = FakeModel()
    error = RuntimeError("backend failed")

    def fail(prepared):
        raise error

    model.infer = fail
    with pytest.raises(RuntimeError) as caught:
        benchmark(model, object(), runs=2, warmup=0)
    assert caught.value is error
    assert model.calls == ["preprocess"]


def test_tensor_contracts_have_no_request_or_timing_state():
    assert {field.name for field in fields(PreparedInput)} == {"tensors", "metadata"}
    assert {field.name for field in fields(RawOutput)} == {"tensors", "prepared"}
