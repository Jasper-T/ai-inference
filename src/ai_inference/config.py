"""Validated YAML configuration. Tensor shapes always include the batch axis."""

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelConfig(Section):
    name: str = Field(min_length=1)
    path: Path


class BatchConfig(Section):
    mode: Literal["fixed", "dynamic"] = "fixed"
    size: int = Field(default=4, gt=0, strict=True)
    padding: Literal["repeat_last", "zeros"] = "repeat_last"


class RuntimeConfig(Section):
    backend: str = "onnxruntime"
    warmup: bool = Field(default=True, strict=True)
    warmup_runs: int = Field(default=3, gt=0, strict=True)
    batch: BatchConfig = Field(default_factory=BatchConfig)
    # Each engine validates its own strict options schema.
    options: dict[str, Any] = Field(default_factory=dict)


class TensorSpec(Section):
    name: str = Field(min_length=1)
    dtype: Literal["float32", "float16", "float64", "int64", "int32", "uint8", "bool"] | None = None
    shape: tuple[int | None, ...] | None = None

    @model_validator(mode="after")
    def check_shape(self):
        if self.shape is not None and (
            not self.shape or any(x is not None and x <= 0 for x in self.shape)
        ):
            raise ValueError("shape must have positive dimensions or null for dynamic dimensions")
        return self


class Config(Section):
    model: ModelConfig
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    inputs: tuple[TensorSpec, ...] | None = Field(default=None, min_length=1)
    outputs: tuple[TensorSpec, ...] | None = Field(default=None, min_length=1)
    # Ordered Pipeline operations; factories validate their own strict options.
    preprocess: dict[str, Any]
    postprocess: dict[str, Any]

    @model_validator(mode="after")
    def check_contract(self):
        if "steps" not in self.preprocess or "steps" not in self.postprocess:
            raise ValueError(
                "preprocess and postprocess must both use ordered steps; flat parameters are unsupported"
            )
        batch = self.runtime.batch
        for specs in (self.inputs, self.outputs):
            if specs is None:
                continue
            if len({s.name for s in specs}) != len(specs):
                raise ValueError("duplicate tensor names")
            for spec in specs:
                expected = batch.size if batch.mode == "fixed" else None
                if spec.shape is not None and spec.shape[0] != expected:
                    raise ValueError(
                        "all tensor batch dimensions must match runtime.batch (null for dynamic)"
                    )
        if self.runtime.backend not in ("onnxruntime", "om") and not self.has_tensor_contract:
            raise ValueError("this backend requires explicit inputs/outputs with dtype and shape")
        return self

    @property
    def has_tensor_contract(self) -> bool:
        return all(
            specs is not None and all(s.dtype is not None and s.shape is not None for s in specs)
            for specs in (self.inputs, self.outputs)
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        path = Path(path).expanduser().resolve()
        config = cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
        model_path = config.model.path.expanduser()
        if not model_path.is_absolute():
            model_path = path.parent / model_path
        return config.model_copy(
            update={"model": config.model.model_copy(update={"path": model_path.resolve()})}
        )
