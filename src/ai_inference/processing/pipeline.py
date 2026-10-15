"""Compile ordered steps before runtime allocation; execute request-local contexts."""

from dataclasses import dataclass
from typing import Any, Callable, overload

from pydantic import Field

from ai_inference.config import Config, Section
from ai_inference.contracts.data import (
    FaceInput,
    FeaturePrediction,
    ModelInput,
    Prediction,
    PreparedInput,
    RawOutput,
    Result,
)
from ai_inference.contracts.registry import Registry
from ai_inference.contracts.validation import validate_tensors
from ai_inference.logging import log_event
from ai_inference.processing.common import image_list
from ai_inference.processing.context import Context, Geometry, SampleContext


class PipelineConfig(Section):
    device: str = Field(default="cpu", pattern="^(cpu|cuda|trt)$")
    source_color: str = Field(default="BGR", pattern="^(BGR|RGB)$")
    steps: tuple[dict[str, dict[str, Any]], ...] = Field(min_length=1)


@dataclass
class State:
    kind: str
    layout: str = "HWC"
    color: str = "BGR"
    dtype: str = "uint8"
    height: int | None = None
    width: int | None = None
    transformed: bool = False
    restorable: bool = True
    uniform: bool = True
    padded: bool = False


@dataclass(frozen=True)
class Step:
    """Factory-built operation, with compile-time plan and explicit device support."""

    input_kind: str
    output_kind: str
    run: Callable[[Context], None]
    plan: Callable[[State], None]
    devices: tuple[str, ...] = ("cpu",)
    required_color: str | None = None
    geometry_mode: str | None = None


StepRegistry = Registry("processing step")


class Pipeline:
    def __init__(self, stage: str, section: dict, config: Config):
        if stage not in ("preprocess", "postprocess"):
            raise ValueError("pipeline stage must be preprocess or postprocess")
        self.stage, self.config = stage, config
        self.settings = PipelineConfig.model_validate(section)
        device = self.settings.device
        if device != "cpu":
            raise ValueError(f"{stage} device {device!r} is not implemented")
        state = State(
            "images" if stage == "preprocess" else "raw", color=self.settings.source_color
        )
        self.steps = []
        for index, entry in enumerate(self.settings.steps):
            if len(entry) != 1:
                raise ValueError(f"{stage} step {index}: exactly one operation required")
            name, params = next(iter(entry.items()))
            params = dict(params)
            placement = params.pop("device", device)
            if placement != "cpu":
                raise ValueError(f"{name} device {placement!r} is not implemented")
            step = StepRegistry.create(name, config, **params)
            if not isinstance(step, Step):
                raise TypeError(f"{name}: step factory must return Step")
            if placement not in step.devices:
                raise ValueError(f"{name} device {placement!r} is not implemented")
            if state.kind != step.input_kind:
                raise ValueError(f"{name}: requires {step.input_kind}, received {state.kind}")
            step.plan(state)
            state.kind = step.output_kind
            self.steps.append((name, step))
        expected = "prepared" if stage == "preprocess" else "results"
        self.state = state
        if state.kind != expected:
            raise ValueError(f"{stage} pipeline must end in {expected}, got {state.kind}")

    @overload
    def __call__(self, data: RawOutput) -> list[Result]: ...

    @overload
    def __call__(self, data: ModelInput) -> PreparedInput: ...

    def __call__(self, data):
        if self.stage == "preprocess":
            images = data if isinstance(data, list) else [data]
            samples = []
            for value in images:
                image = value.image if isinstance(value, FaceInput) else value
                image_list(image)
                samples.append(
                    SampleContext(
                        image,
                        Geometry(tuple(image.shape[:2])),
                        self.settings.source_color,
                        keypoints=value.keypoints if isinstance(value, FaceInput) else None,
                    )
                )
            context = Context(samples, "images")
        else:
            if not isinstance(data, RawOutput):
                raise TypeError("postprocess requires RawOutput")
            validate_tensors(data.tensors, self.config.outputs, data.prepared.count)
            context = Context(data, "raw")
        for index, (name, step) in enumerate(self.steps, 1):
            try:
                if context.kind != step.input_kind or context.device != "cpu":
                    raise ValueError(f"{name}: invalid context kind/device")
                step.run(context)
                if context.kind != step.output_kind or context.device != "cpu":
                    raise ValueError(f"{name}: step violated its output kind/device contract")
            except Exception as exc:
                log_event(
                    "infer",
                    f"{self.stage}.{name}",
                    f"Failed | step={index} | error={type(exc).__name__}",
                    level="WARNING",
                    model=self.config.model.name,
                    error_type=type(exc).__name__,
                    step_index=index,
                    status="failed",
                )
                raise
            log_event(
                "infer",
                f"{self.stage}.{name}",
                f"Completed | step={index}",
                level="DEBUG",
                model=self.config.model.name,
                step_index=index,
                status="completed",
            )
        if self.stage == "preprocess":
            if not isinstance(context.data, PreparedInput):
                raise TypeError("preprocess pipeline must produce PreparedInput")
            validate_tensors(context.data.tensors, self.config.inputs, context.data.count)
        else:
            if not isinstance(data, RawOutput):
                raise TypeError("postprocess requires RawOutput")
            if (
                not isinstance(context.data, list)
                or len(context.data) != data.prepared.count
                or any(
                    not isinstance(item, (Prediction, FeaturePrediction)) for item in context.data
                )
            ):
                raise ValueError("postprocess pipeline must return one Result per real sample")
        return context.data


def validate_connection(pre: Pipeline, post: Pipeline) -> None:
    for name, step in post.steps:
        if step.required_color is not None and pre.state.color != step.required_color:
            raise ValueError(f"{name}: model input color must be {step.required_color}")
        if step.geometry_mode is not None and not pre.state.restorable:
            raise ValueError(f"{name}: cannot restore face-aligned coordinates")
        if step.geometry_mode == "uniform" and not pre.state.uniform:
            raise ValueError(f"{name}: requires uniform resize geometry")
        if step.geometry_mode == "resize" and pre.state.padded:
            raise ValueError(f"{name}: requires direct resize without padding")


def step_options(config, name):
    """Retrieve an explicitly configured step for reference/inspection tools."""
    matches = [
        entry[name]
        for section in (config.preprocess, config.postprocess)
        for entry in section.get("steps", ())
        if name in entry
    ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one configured {name} step")
    return {k: v for k, v in matches[0].items() if k != "device"}
