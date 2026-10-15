"""Public model facade: initialize components and orchestrate request stages."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
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


class Model:
    def __init__(
        self,
        config: Config | str | Path,
        *,
        preprocessor: Preprocessor | None = None,
        postprocessor: Postprocessor | None = None,
        logger: Logger | None = None,
    ):
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

        pre, post = compile_pipelines() if self.config.has_tensor_contract else (None, None)
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
        self.batcher = Batcher(self.config)
        self.predictor = DefaultPredictor(self.engine, self.config, batcher=self.batcher)
        if hasattr(self.engine, "log_tensor_metadata"):
            self.engine.log_tensor_metadata(self.logger)
        policy = self.config.runtime.batch
        padding = policy.padding if policy.mode == "fixed" else "none"
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

    def warmup(self, tensors: Mapping[str, np.ndarray] | None = None) -> None:
        """Explicit warmup with optional valid physical-batch tensors."""
        with log_context(self.logger, model=self.config.model.name):
            if tensors is None:
                self.engine.warmup()
            else:
                self.engine.warmup(tensors)

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

    def predict(self, images: ModelInput) -> list[Result]:
        return self.postprocess(self.infer(self.preprocess(images)))
