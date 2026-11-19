"""Public model facade: initialize components and orchestrate request stages."""

from __future__ import annotations

from collections.abc import Mapping
from functools import wraps
from pathlib import Path
from threading import Condition, local
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger as default_logger

from ai_inference.config import Config
from ai_inference.contracts.data import (
    FeaturePrediction,
    ModelInput,
    Prediction,
    PreparedInput,
    RawOutput,
    Result,
)
from ai_inference.contracts.processors import Postprocessor, Preprocessor
from ai_inference.contracts.validation import validate_tensors
from ai_inference.engines.registry import EngineRegistry
from ai_inference.inference.batcher import Batcher
from ai_inference.inference.predictor import DefaultPredictor
from ai_inference.logging import log_context, log_event
from ai_inference.processing import Pipeline
from ai_inference.processing.pipeline import validate_connection

if TYPE_CHECKING:
    from loguru import Logger


def _open_operation(method):
    @wraps(method)
    def checked(self, *args, **kwargs):
        if getattr(self._operation_local, "active", False):
            return method(self, *args, **kwargs)
        with self._lifecycle_condition:
            if self._closed or self._closing:
                raise RuntimeError("Model is closed")
            self._active_operations += 1
        self._operation_local.active = True
        try:
            return method(self, *args, **kwargs)
        finally:
            self._operation_local.active = False
            with self._lifecycle_condition:
                self._active_operations -= 1
                self._lifecycle_condition.notify_all()

    return checked


class Model:
    def __init__(
        self,
        config: Config | str | Path,
        *,
        preprocessor: Preprocessor | None = None,
        postprocessor: Postprocessor | None = None,
        logger: Logger | None = None,
    ):
        self._lifecycle_condition = Condition()
        self._operation_local = local()
        self._active_operations = 0
        self._closing = False
        self._closed = False
        self.config = (
            config.model_copy(deep=True) if isinstance(config, Config) else Config.from_yaml(config)
        )
        self.logger = (default_logger if logger is None else logger).bind(
            component="ai_inference.model"
        )
        with log_context(self.logger, model=self.config.model.name):
            try:
                self._initialize(preprocessor, postprocessor)
            except Exception as exc:
                try:
                    self.close()
                except Exception:
                    log_event("init", "model.cleanup", "Resource cleanup failed", level="WARNING")
                log_event(
                    "init",
                    "model.failed",
                    f"Initialization failed | error={type(exc).__name__}",
                    level="WARNING",
                    status="failed",
                    error_type=type(exc).__name__,
                )
                raise
            log_event(
                "init",
                "model.ready",
                f"Ready | backend={self.config.runtime.backend}",
                status="completed",
                backend=self.config.runtime.backend,
            )

    def _initialize(
        self, preprocessor: Preprocessor | None, postprocessor: Postprocessor | None
    ) -> None:
        if "steps" not in self.config.preprocess or "steps" not in self.config.postprocess:
            raise ValueError("Model requires preprocess/postprocess.steps")

        # Explicit contracts can still reject invalid pipelines before loading.
        def compile_pipelines() -> tuple[Pipeline, Pipeline]:
            try:
                pre = Pipeline("preprocess", self.config.preprocess, self.config)
                post = Pipeline("postprocess", self.config.postprocess, self.config)
                validate_connection(pre, post)
            except Exception as exc:
                log_event(
                    "init",
                    "pipeline.compile",
                    f"Failed | error={type(exc).__name__}",
                    level="WARNING",
                    error_type=type(exc).__name__,
                    status="failed",
                )
                raise
            return pre, post

        pre, post = (
            compile_pipelines()
            if self.config.has_tensor_contract and self.config.runtime.batch.mode != "auto"
            else (None, None)
        )
        try:
            self.engine = EngineRegistry.create(self.config.runtime.backend, self.config)
        except Exception as exc:
            log_event(
                "init",
                "engine.load",
                f"Failed | error={type(exc).__name__}",
                level="WARNING",
                error_type=type(exc).__name__,
                status="failed",
            )
            raise
        resolved = self.engine.config
        changed = resolved != self.config
        self.config = resolved
        providers = getattr(self.engine, "selected_providers", ())
        message = f"Loaded | backend={self.config.runtime.backend}"
        if providers:
            message += f" | providers={', '.join(providers)}"
        log_event(
            "init",
            "engine.load",
            message,
            backend=self.config.runtime.backend,
            providers=list(providers),
            status="completed",
        )
        if pre is None or post is None or changed:
            pre, post = compile_pipelines()
        self.preprocessor: Preprocessor = pre if preprocessor is None else preprocessor
        self.postprocessor: Postprocessor = post if postprocessor is None else postprocessor
        self.batcher = Batcher(
            self.config, capabilities=getattr(self.engine, "batch_capabilities", None)
        )
        self.predictor = DefaultPredictor(self.engine, self.config, batcher=self.batcher)
        if hasattr(self.engine, "log_tensor_metadata"):
            self.engine.log_tensor_metadata(self.logger)
        policy = self.config.runtime.batch
        padding = policy.padding if policy.mode == "fixed" or self.batcher.capabilities else "none"
        size_role = "fixed" if policy.mode == "fixed" else "max"
        log_event(
            "init",
            "batching",
            f"Validated | mode={policy.mode} | size={policy.size} | size_role={size_role} | padding={padding}",
            batch_mode=policy.mode,
            batch_size=policy.size,
            batch_size_role=size_role,
            batch_padding=padding,
            status="validated",
        )
        if self.config.runtime.warmup:
            self.engine.warmup()

    def close(self) -> None:
        """Wait for model operations and release owned engine resources once."""
        if getattr(self._operation_local, "active", False):
            raise RuntimeError("Cannot close Model from an active model operation")
        with self._lifecycle_condition:
            while self._closing and not self._closed:
                self._lifecycle_condition.wait()
            if self._closed:
                return
            self._closing = True
            while self._active_operations:
                self._lifecycle_condition.wait()
            engine = getattr(self, "engine", None)
            close = getattr(engine, "close", None)
            try:
                if close is not None:
                    close()
            finally:
                # Legacy backends have no explicit close contract; relinquish our ownership.
                self.engine = None
                self.predictor = None
                self._closed = True
                self._lifecycle_condition.notify_all()

    def __enter__(self):
        with self._lifecycle_condition:
            if self._closed or self._closing:
                raise RuntimeError("Model is closed")
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()

    @_open_operation
    def warmup(self, tensors: Mapping[str, np.ndarray] | None = None) -> None:
        """Explicit warmup with optional valid physical-batch tensors."""
        with log_context(self.logger, model=self.config.model.name):
            if tensors is None:
                self.engine.warmup()
            else:
                self.engine.warmup(tensors)

    @_open_operation
    def preprocess(self, images: ModelInput) -> PreparedInput:
        with log_context(self.logger, model=self.config.model.name):
            try:
                prepared = self.preprocessor(images)
                if not isinstance(prepared, PreparedInput):
                    raise TypeError("preprocessor must return PreparedInput")
                validate_tensors(prepared.tensors, self.config.inputs, prepared.count)
            except Exception as exc:
                log_event(
                    "infer",
                    "preprocess",
                    f"Failed | error={type(exc).__name__}",
                    level="WARNING",
                    failed_stage="preprocess",
                    error_type=type(exc).__name__,
                    status="failed",
                )
                raise
            log_event(
                "infer",
                "preprocess",
                f"Completed | images={prepared.count}",
                sample_count=prepared.count,
                status="completed",
            )
            return prepared

    @_open_operation
    def infer(self, prepared: PreparedInput) -> RawOutput:
        with log_context(self.logger, model=self.config.model.name):
            try:
                if not isinstance(prepared, PreparedInput):
                    raise TypeError("infer requires PreparedInput")
                validate_tensors(prepared.tensors, self.config.inputs, prepared.count)
                raw = self.predictor(prepared)
                if not isinstance(raw, RawOutput) or raw.prepared is not prepared:
                    raise TypeError(
                        "predictor must return RawOutput carrying the same PreparedInput"
                    )
                validate_tensors(raw.tensors, self.config.outputs, prepared.count)
            except Exception as exc:
                log_event(
                    "infer",
                    "predict",
                    f"Failed | error={type(exc).__name__}",
                    level="WARNING",
                    failed_stage="predict",
                    error_type=type(exc).__name__,
                    status="failed",
                )
                raise
            log_event(
                "infer",
                "predict",
                f"Completed | images={prepared.count}",
                sample_count=prepared.count,
                status="completed",
            )
            return raw

    @_open_operation
    def postprocess(self, outputs: RawOutput) -> list[Result]:
        if not isinstance(outputs, RawOutput):
            raise TypeError("postprocess requires RawOutput")
        with log_context(self.logger, model=self.config.model.name):
            try:
                validate_tensors(outputs.tensors, self.config.outputs, outputs.prepared.count)
                results = self.postprocessor(outputs)
                if (
                    not isinstance(results, list)
                    or len(results) != outputs.prepared.count
                    or any(
                        not isinstance(item, (Prediction, FeaturePrediction)) for item in results
                    )
                ):
                    raise ValueError("postprocessor must return one Result per real sample")
                result_count = len(results)
            except Exception as exc:
                log_event(
                    "infer",
                    "postprocess",
                    f"Failed | error={type(exc).__name__}",
                    level="WARNING",
                    failed_stage="postprocess",
                    error_type=type(exc).__name__,
                    status="failed",
                )
                raise
            log_event(
                "infer",
                "postprocess",
                f"Completed | results={result_count}",
                result_count=result_count,
                status="completed",
            )
            return results

    @_open_operation
    def predict(self, images: ModelInput) -> list[Result]:
        return self.postprocess(self.infer(self.preprocess(images)))
