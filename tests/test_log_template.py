import numpy as np
import pytest
from loguru import logger
from test_ort import fake_ort
from test_tensor_boundaries import RecorderEngine

from ai_inference import Config, Model, Plotter, log_context, log_event


@pytest.fixture
def events():
    captured = []
    sink = logger.add(lambda message: captured.append(message.record), level="DEBUG")
    try:
        yield captured
    finally:
        logger.remove(sink)


@pytest.fixture
def model(config, monkeypatch):
    monkeypatch.setattr(
        "ai_inference.model.EngineRegistry.create", lambda backend, config: RecorderEngine(config)
    )
    return Model(config)


def test_initialization_template_and_warmup(config, monkeypatch, events):
    monkeypatch.setattr(
        "ai_inference.model.EngineRegistry.create", lambda backend, config: RecorderEngine(config)
    )
    data = config.model_dump()
    data["runtime"].update(warmup=True, warmup_runs=2)
    model = Model(Config.model_validate(data))
    steps = [record["extra"]["step"] for record in events]
    assert steps == [
        "engine.load",
        "engine.metadata",
        "engine.metadata",
        "engine.metadata",
        "batching",
        "engine.warmup",
        "model.ready",
    ]
    assert all(record["message"].startswith("[rfdetr_synthetic]-[init]-[") for record in events)
    assert all(record["extra"].get("request_id") is None for record in events)
    assert "Input 'input' | float32 | shape=[4, 3, 560, 560]" in events[1]["message"]
    assert model.engine.calls == [4, 4]


@pytest.mark.parametrize(
    "mode,padding,effective_padding,size_role",
    [
        ("fixed", "repeat_last", "repeat_last", "fixed"),
        ("fixed", "zeros", "zeros", "fixed"),
        ("dynamic", "repeat_last", "none", "max"),
    ],
)
def test_validated_batch_and_resolved_io_logging(
    config, monkeypatch, tmp_path, events, mode, padding, effective_padding, size_role
):
    import sys

    explicit, _ = fake_ort(monkeypatch, tmp_path, config)
    session = sys.modules["onnxruntime"].InferenceSession
    for method in ("get_inputs", "get_outputs"):
        original = getattr(session, method)

        def metadata(self, original=original):
            nodes = original(self)
            for node in nodes:
                node.shape = ["N", *node.shape[1:]]
            return nodes

        monkeypatch.setattr(session, method, metadata)
    data = explicit.model_dump()
    data.pop("inputs")
    data.pop("outputs")
    data["runtime"]["batch"].update(mode=mode, padding=padding)
    app_logger = logger.bind(application="validated-config")
    model = Model(Config.model_validate(data), logger=app_logger)
    captured = [r for r in events if r["extra"].get("application") == "validated-config"]
    batch_records = [r for r in captured if r["extra"]["step"] == "batching"]
    assert len(batch_records) == 1
    batch = batch_records[0]
    assert batch["level"].name == "INFO"
    assert batch["extra"]["stage"] == "init"
    assert batch["extra"]["batch_mode"] == mode
    assert batch["extra"]["batch_size"] == 4
    assert batch["extra"]["batch_size_role"] == size_role
    assert batch["extra"]["batch_padding"] == effective_padding
    assert batch["extra"]["status"] == "validated"
    assert f"padding={effective_padding}" in batch["message"]
    records = [r for r in captured if r["extra"]["step"] == "engine.metadata"]
    assert len(records) == 3
    specs = {s.name: s for s in (*model.config.inputs, *model.config.outputs)}
    for record in records:
        fields = record["extra"]
        assert fields["shape"][0] == "N"
        assert fields["resolved_shape"] == list(specs[fields["tensor_name"]].shape)
        assert fields["resolved_shape"][0] == (4 if mode == "fixed" else None)
        assert "resolved_shape=" in record["message"]


def test_invalid_batch_has_no_validated_summary(config, monkeypatch, tmp_path, events):
    explicit, _ = fake_ort(monkeypatch, tmp_path, config)
    data = explicit.model_dump()
    data.pop("inputs")
    data.pop("outputs")
    data["runtime"]["batch"]["size"] = 2
    with pytest.raises(ValueError, match="model batch=4"):
        Model(Config.model_validate(data))
    assert not any(r["extra"]["step"] == "batching" for r in events)
    assert not any(r["extra"]["step"] == "model.ready" for r in events)


def test_application_and_visualization_share_context(tmp_path, events):
    events.clear()
    with log_context(model="rfdetr_synthetic", request_id="camera-frame", scene="camera-1"):
        log_event("business", "alarm.evaluate", "No alarm", alarm_count=0)
        Plotter().save(tmp_path / "annotation.png", np.zeros((10, 20, 3), np.uint8))
    log_event("business", "idle", "No active context")
    assert events[0]["message"].startswith("[rfdetr_synthetic]-[business]-[alarm.evaluate]")
    assert events[0]["function"] == "test_application_and_visualization_share_context"
    assert events[1]["message"].startswith("[plotter]-[save] Saved | path=")
    assert events[1]["function"] == "save"
    assert events[1]["extra"]["component"] == "plotter"
    assert events[1]["extra"]["model"] == "rfdetr_synthetic"
    assert events[1]["extra"]["step"] == "plotter.save"
    assert events[1]["extra"]["stage"] == "visualize"
    assert events[0]["extra"]["request_id"] == events[1]["extra"]["request_id"] == "camera-frame"
    assert events[1]["extra"]["scene"] == "camera-1"
    assert events[2]["extra"].get("request_id") is None
    assert events[2]["extra"]["model"] == "-"


def test_plotter_logging_without_model_context(tmp_path, events):
    Plotter().save(tmp_path / "annotation.png", np.zeros((10, 20, 3), np.uint8))
    assert len(events) == 1
    assert events[0]["message"].startswith("[plotter]-[save] Saved | path=")
    assert "model" not in events[0]["extra"]
    assert events[0]["extra"]["status"] == "completed"


def test_plotter_failure_preserves_application_logger(tmp_path, monkeypatch, events):
    monkeypatch.setattr("cv2.imwrite", lambda *args: False)
    path = tmp_path / "annotation.png"
    with log_context(logger=logger.bind(application="cv"), request_id="frame-001"):
        with pytest.raises(OSError, match="cannot write annotation"):
            Plotter().save(path, np.zeros((10, 20, 3), np.uint8))
    assert len(events) == 1
    record = events[0]
    assert record["message"] == "[plotter]-[save] Failed | error=OSError"
    assert record["level"].name == "WARNING"
    assert record["function"] == "save"
    assert record["extra"]["application"] == "cv"
    assert record["extra"]["request_id"] == "frame-001"
    assert record["extra"]["status"] == "failed"
    assert record["extra"]["error_type"] == "OSError"
    assert record["extra"]["path"] == str(path)


def test_warmup_failure_is_attributed_to_initialization(config, monkeypatch, events):
    class FailingEngine(RecorderEngine):
        def _infer(self, tensors):
            raise RuntimeError("private-model-input")

    monkeypatch.setattr(
        "ai_inference.model.EngineRegistry.create", lambda backend, config: FailingEngine(config)
    )
    data = config.model_dump()
    data["runtime"]["warmup"] = True
    with pytest.raises(RuntimeError, match="private-model-input"):
        Model(Config.model_validate(data))
    failures = [record for record in events if record["level"].name == "WARNING"]
    assert [record["extra"]["step"] for record in failures] == ["engine.warmup", "model.failed"]
    assert all(record["extra"]["stage"] == "init" for record in failures)
    assert all("private-model-input" not in record["message"] for record in events)
    assert not any(record["extra"]["step"] == "model.ready" for record in events)
