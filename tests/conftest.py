import logging
from pathlib import Path

import pytest
from loguru import logger
from model_contracts import load_synthetic_config

from ai_inference import Config


@pytest.fixture(autouse=True)
def capture_loguru(caplog):
    """Capture native Loguru records with pytest's existing caplog assertions."""

    def sink(message):
        record = message.record
        captured = logging.makeLogRecord(
            {
                **record["extra"],
                "name": record["extra"].get("component", record["name"]),
                "levelno": record["level"].no,
                "levelname": record["level"].name,
                "msg": record["message"],
                "args": (),
            }
        )
        caplog.handler.handle(captured)

    sink_id = logger.add(sink, level="INFO")
    yield
    logger.remove(sink_id)


@pytest.fixture
def config():
    loaded = load_synthetic_config(
        Path(__file__).resolve().parent / "fixtures/rfdetr_synthetic.yaml"
    )
    data = loaded.model_dump()
    data["runtime"]["options"]["providers"] = "cuda"
    data["runtime"]["warmup"] = False
    return Config.model_validate(data)
