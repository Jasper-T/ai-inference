from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping, Protocol

import numpy as np

from ai_inference.config import Config, TensorSpec
from ai_inference.contracts.validation import validate_physical_inputs, validate_tensors
from ai_inference.logging import log_context, log_event

if TYPE_CHECKING:
    from loguru import Logger


@dataclass(frozen=True)
class TensorMetadata:
    name: str
    dtype: str
    shape: tuple[int | str | None, ...]


class Engine(Protocol):
    """Execute one physical batch, preserving names, dtype and leading batch axis."""

    config: Config

    def infer(self, tensors: Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]: ...
    def warmup(self, tensors: Mapping[str, np.ndarray] | None = None) -> None: ...


class BaseEngine:
    """Checked public inference entry; subclasses implement backend execution only."""

    def __init__(self, config: Config):
        self.config = config

    def read_tensor_metadata(self) -> dict[str, tuple[TensorMetadata, ...]]:
        """Return backend-neutral metadata; backends may inspect their session."""
        return {
            kind: tuple(
                TensorMetadata(s.name, s.dtype, s.shape) for s in getattr(self.config, kind)
            )
            for kind in ("inputs", "outputs")
        }

    def _load_tensor_contract(self) -> None:
        """Resolve optional YAML constraints in the backend's actual tensor order."""
        data = self.config.model_dump()
        for kind, tensors in self.read_tensor_metadata().items():
            data[kind] = self._resolve_metadata(tensors, getattr(self.config, kind), kind)
        self.config = Config.model_validate(data)

    def _resolve_metadata(self, actual, specs, kind):
        nodes = {n.name: n for n in actual}
        if not actual or len(nodes) != len(actual):
            raise ValueError(
                f"{self.config.runtime.backend} {kind} metadata must have unique tensor names"
            )
        if specs is not None and set(nodes) != {s.name for s in specs}:
            raise ValueError(
                f"{self.config.runtime.backend} {kind} names do not match YAML contract"
            )
        expected = {s.name: s for s in specs or ()}
        resolved = []
        for node in actual:
            dtype = node.dtype
            shape = tuple(d if isinstance(d, int) else None for d in node.shape)
            policy = self.config.runtime.batch
            if shape:
                model_batch = shape[0]
                if model_batch is not None and (
                    policy.mode == "dynamic" or model_batch != policy.size
                ):
                    raise ValueError(
                        f"{self.config.runtime.backend} {kind} tensor '{node.name}': "
                        f"model batch={model_batch} conflicts with "
                        f"runtime.batch.mode={policy.mode}, size={policy.size}; "
                        f"use mode=fixed, size={model_batch} for this static model"
                    )
                if model_batch is None and policy.mode == "fixed":
                    shape = (policy.size, *shape[1:])
            spec = expected.get(node.name)
            if spec is not None:
                if spec.dtype is not None and spec.dtype != dtype:
                    raise ValueError(
                        f"{self.config.runtime.backend} {kind} dtype mismatch: {node.name}"
                    )
                if spec.shape is not None:
                    if len(shape) != len(spec.shape):
                        raise ValueError(
                            f"{self.config.runtime.backend} {kind} rank mismatch: {node.name}"
                        )
                    for actual_dim, configured in zip(shape, spec.shape):
                        if actual_dim is not None and actual_dim != configured:
                            raise ValueError(
                                f"{self.config.runtime.backend} {kind} shape mismatch: {node.name}"
                            )
                    # Concrete constraints on symbolic dimensions remain runtime checks.
                    shape = spec.shape
            resolved.append(TensorSpec(name=node.name, dtype=dtype, shape=shape))
        return resolved

    def log_tensor_metadata(self, logger: Logger) -> None:
        for kind, tensors in self.read_tensor_metadata().items():
            resolved = {s.name: s for s in getattr(self.config, kind)}
            for tensor in tensors:
                resolved_shape = list(resolved[tensor.name].shape)
                with log_context(logger, model=self.config.model.name):
                    direction = "Input" if kind == "inputs" else "Output"
                    log_event(
                        "init",
                        "engine.metadata",
                        f"{direction} '{tensor.name}' | {tensor.dtype} | shape={list(tensor.shape)} | resolved_shape={resolved_shape}",
                        kind=kind,
                        tensor_name=tensor.name,
                        dtype=tensor.dtype,
                        shape=list(tensor.shape),
                        resolved_shape=resolved_shape,
                    )

    def create_warmup_inputs(self) -> Mapping[str, np.ndarray]:
        """Default zero inputs; override for model-specific values or shapes."""
        batch = self.config.runtime.batch.size if self.config.runtime.batch.mode == "fixed" else 1
        tensors = {}
        specs = self.config.inputs
        if specs is None:
            raise ValueError("warmup requires a resolved input tensor contract")
        for spec in specs:
            shape, dtype = spec.shape, spec.dtype
            if shape is None or dtype is None:
                raise ValueError("warmup requires a resolved input tensor contract")
            dimensions = []
            for dim in shape[1:]:
                if dim is None:
                    raise ValueError(
                        "warmup requires concrete non-batch input dimensions; provide tensors, specify sizes or set runtime.warmup=false"
                    )
                dimensions.append(dim)
            tensors[spec.name] = np.zeros((batch, *dimensions), dtype=dtype)
        return tensors

    def warmup(self, tensors: Mapping[str, np.ndarray] | None = None) -> None:
        """Create or accept valid physical inputs, then run the engine entry."""
        completed = 0
        try:
            tensors = self.create_warmup_inputs() if tensors is None else tensors
            batch = validate_physical_inputs(tensors, self.config)
            for _ in range(self.config.runtime.warmup_runs):
                self.infer(tensors)
                completed += 1
        except Exception as exc:
            log_event(
                "init",
                "engine.warmup",
                f"Failed | completed runs={completed} | error={type(exc).__name__}",
                level="WARNING",
                model=self.config.model.name,
                error_type=type(exc).__name__,
                completed_runs=completed,
                status="failed",
            )
            raise
        log_event(
            "init",
            "engine.warmup",
            f"Completed | runs={self.config.runtime.warmup_runs} | batch={batch}",
            model=self.config.model.name,
            warmup_runs=self.config.runtime.warmup_runs,
            physical_batch=batch,
            status="completed",
        )

    def infer(self, tensors: Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
        count = validate_physical_inputs(tensors, self.config)
        outputs = self._infer(tensors)
        if not isinstance(outputs, Mapping):
            raise ValueError("outputs must be a tensor mapping")
        validate_tensors(outputs, self.config.outputs, count)
        return outputs

    def _infer(self, tensors: Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
        raise NotImplementedError
