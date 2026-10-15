"""Readable log templates and application-provided context; no request statistics."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING

from loguru import logger as default_logger

if TYPE_CHECKING:
    from loguru import Logger


LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <7} | {message}"


_context: ContextVar[tuple[Logger, dict] | None] = ContextVar(
    "ai_inference_log_context", default=None
)


@contextmanager
def log_context(logger: Logger | None = None, **fields):
    """Carry correlation fields through nested SDK and application operations."""
    current = _context.get()
    inherited_logger, inherited_fields = current if current is not None else (default_logger, {})
    token = _context.set(
        (logger if logger is not None else inherited_logger, {**inherited_fields, **fields})
    )
    try:
        yield
    finally:
        _context.reset(token)


def log_event(stage: str, step: str, message: str, *, level: str = "INFO", **fields) -> None:
    """Use [model]-[stage]-[step], keeping all fields available to JSON sinks."""
    current = _context.get()
    logger, context = current if current is not None else (default_logger, {})
    values = {"model": "-", **context, **fields, "stage": stage, "step": step}
    identifiers = []
    if values.get("batch_index") is not None:
        identifiers.append(f"batch={values['batch_index']}/{values['batch_count']}")
    prefix = f"[{values['model']}]-[{stage}]-[{step}]"
    correlation = " ".join(identifiers)
    logger.bind(**values).opt(depth=1).log(
        level, "{} {}{}{}", prefix, correlation, " | " if correlation else "", message
    )


def log_component_event(
    component: str, operation: str, message: str, *, level: str = "INFO", **fields
) -> None:
    """Log model-independent operations while preserving application context."""
    current = _context.get()
    logger, context = current if current is not None else (default_logger, {})
    values = {
        **context,
        **fields,
        "component": component,
        "step": f"{component}.{operation}",
    }
    logger.bind(**values).opt(depth=1).log(level, "[{}]-[{}] {}", component, operation, message)
