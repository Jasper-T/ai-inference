"""Logical batch prediction, independent of model algorithms and image processing."""

from typing import Protocol

from ai_inference.config import Config
from ai_inference.contracts.data import PreparedInput, RawOutput
from ai_inference.engines.base import Engine
from ai_inference.inference.batcher import Batcher
from ai_inference.logging import log_context, log_event


class Predictor(Protocol):
    def __call__(self, prepared: PreparedInput) -> RawOutput: ...


class DefaultPredictor:
    """Run every request through the same Batcher and physical Engine boundary."""

    def __init__(self, engine: Engine, config: Config, *, batcher: Batcher | None = None):
        self.engine = engine
        self.config = config
        self.batcher = Batcher(config) if batcher is None else batcher
        if self.batcher.config != config:
            raise ValueError("Batcher and Predictor contracts must match")

    def __call__(self, prepared: PreparedInput) -> RawOutput:
        parts = []
        for chunk in self.batcher.split(prepared):
            with log_context(
                model=self.config.model.name, batch_index=chunk.index, batch_count=chunk.total
            ):
                try:
                    outputs = self.engine.infer(chunk.tensors)
                    # Also checks custom engines that don't inherit BaseEngine.
                    trimmed = self.batcher.trim(outputs, chunk)
                except Exception as exc:
                    log_event(
                        "infer",
                        "engine.predict",
                        f"Failed | error={type(exc).__name__}",
                        level="WARNING",
                        error_type=type(exc).__name__,
                        status="failed",
                        physical_batch=chunk.physical_count,
                        valid_samples=chunk.valid_count,
                        padded_samples=chunk.padded_count,
                    )
                    raise
                log_event(
                    "infer",
                    "engine.predict",
                    f"Completed | valid={chunk.valid_count} | physical={chunk.physical_count} | padding={chunk.padded_count}",
                    level="DEBUG",
                    physical_batch=chunk.physical_count,
                    valid_samples=chunk.valid_count,
                    padded_samples=chunk.padded_count,
                    status="completed",
                )
                parts.append(trimmed)
        return self.batcher.merge(parts, prepared)
