import numpy as np
import pytest
from loguru import logger

from ai_inference import Model, PreparedInput, RawOutput
from ai_inference.config import TensorSpec
from ai_inference.contracts.validation import validate_tensors
from ai_inference.engines.base import BaseEngine


class RecorderEngine(BaseEngine):
    def __init__(self, config):
        super().__init__(config)
        self.calls = []

    def _infer(self, tensors):
        self.calls.append(tensors["input"].shape[0])
        count = tensors["input"].shape[0]
        return output_tensors(count)


@pytest.mark.parametrize(
    "specs",
    [
        None,
        (TensorSpec(name="x"),),
        (TensorSpec(name="x", dtype="float32"),),
        (TensorSpec(name="x", shape=(1, 2)),),
    ],
)
def test_unresolved_tensor_contract_is_rejected(specs):
    with pytest.raises(ValueError, match="tensor contract must be resolved"):
        validate_tensors({"x": np.zeros((1, 2), np.float32)}, specs, 1)


def output_tensors(count):
    return {
        "dets": np.zeros((count, 300, 4), np.float32),
        "labels": np.zeros((count, 300, 2), np.float32),
    }


@pytest.fixture
def model(config, monkeypatch):
    monkeypatch.setattr(
        "ai_inference.model.EngineRegistry.create", lambda backend, config: RecorderEngine(config)
    )
    return Model(config)


@pytest.mark.parametrize(
    "tensors",
    [
        {"wrong": np.zeros((1, 3, 560, 560), np.float32)},
        {"input": np.zeros((1, 3, 560, 560), np.float16)},
        {"input": np.zeros((1, 560, 560), np.float32)},
        {"input": np.zeros((1, 3, 320, 320), np.float32)},
        {"input": np.zeros((2, 3, 560, 560), np.float32)},
    ],
)
def test_custom_preprocessor_rejected_before_inference(model, tensors, caplog):
    model.preprocessor = lambda images: PreparedInput(tensors, (object(),))
    with pytest.raises(ValueError):
        model.predict(np.zeros((10, 20, 3), np.uint8))
    assert model.engine.calls == []
    assert [r.failed_stage for r in caplog.records if hasattr(r, "failed_stage")] == ["preprocess"]


def test_external_prepared_rejected_before_backend(model, caplog):
    malformed = PreparedInput({"input": np.zeros((1, 3, 20, 20), np.float32)}, (object(),))
    with pytest.raises(ValueError):
        model.infer(malformed)
    assert model.engine.calls == []
    assert [r.failed_stage for r in caplog.records if hasattr(r, "failed_stage")] == ["predict"]


@pytest.mark.parametrize(
    "malformed",
    [
        {"dets": np.zeros((1, 300, 5), np.float32), "labels": np.zeros((1, 300, 2), np.float32)},
        {"dets": np.zeros((2, 300, 4), np.float32), "labels": np.zeros((1, 300, 2), np.float32)},
        {"dets": np.zeros((1, 300, 4), np.float16), "labels": np.zeros((1, 300, 2), np.float32)},
    ],
)
def test_invalid_output_rejected_before_custom_postprocessor(model, malformed, caplog):
    prepared = model.preprocess(np.zeros((10, 20, 3), np.uint8))
    called = []
    model.postprocessor = lambda raw: called.append(raw)
    with pytest.raises(ValueError):
        model.postprocess(RawOutput(malformed, prepared))
    assert called == []
    assert [r.failed_stage for r in caplog.records if hasattr(r, "failed_stage")] == ["postprocess"]


def test_custom_predictor_invalid_output_rejected_at_predict_boundary(model, caplog):
    prepared = model.preprocess(np.zeros((10, 20, 3), np.uint8))
    malformed = output_tensors(1)
    malformed["dets"] = np.zeros((1, 300, 5), np.float32)
    model.predictor = lambda prepared: RawOutput(malformed, prepared)
    with pytest.raises(ValueError):
        model.infer(prepared)
    assert [r.failed_stage for r in caplog.records if hasattr(r, "failed_stage")] == ["predict"]


def test_logical_count_can_exceed_fixed_physical_batch(model):
    assert len(model.predict([np.zeros((10, 20, 3), np.uint8)] * 5)) == 5
    assert model.engine.calls == [4, 4]


def test_native_loguru_sink_receives_request_fields(model):
    messages = []
    sink = logger.add(
        messages.append, filter=lambda record: record["extra"].get("application") == "boundaries"
    )
    try:
        model.logger = logger.bind(application="boundaries")
        model.predict([])
    finally:
        logger.remove(sink)
    assert messages
    for message in messages:
        record = message.record
        assert "preprocess_time" not in record["extra"]
        assert "request_id" not in record["extra"]
    assert any(message.record["extra"]["step"] == "postprocess" for message in messages)
