"""Reusable CPU image steps; geometry is composed alongside pixel operations."""

import math
from dataclasses import replace
from typing import Literal

import cv2
import numpy as np
from pydantic import Field, model_validator

from ai_inference.config import Section
from ai_inference.contracts.data import PreparedInput
from ai_inference.processing.common import letterbox
from ai_inference.processing.decoders.sface import align_face
from ai_inference.processing.pipeline import Step, StepRegistry


def image_step(plan, operation):
    def run(context):
        for sample in context.data:
            if sample.device != "cpu":
                raise ValueError("CPU steps require CPU sample data")
            operation(sample)

    return Step("images", "images", run, plan)


def require_hwc(state):
    if state.layout != "HWC" or state.dtype != "uint8":
        raise ValueError("image geometry steps require HWC uint8 data")


class ResizeOptions(Section):
    width: int = Field(gt=0, strict=True)
    height: int = Field(gt=0, strict=True)
    keep_ratio: bool = False
    scale_up: bool = True
    backend: Literal["opencv", "pillow"] = "opencv"


@StepRegistry.register("resize")
def resize_step(config, **options):
    p = ResizeOptions.model_validate(options)

    def plan(state):
        require_hwc(state)
        state.height, state.width = (None, None) if p.keep_ratio else (p.height, p.width)
        state.transformed = True
        state.uniform = state.uniform and p.keep_ratio

    def run(sample):
        h, w = sample.data.shape[:2]
        gain = min(p.width / w, p.height / h)
        if not p.scale_up:
            gain = min(gain, 1.0)
        nw, nh = (
            (max(1, min(p.width, round(w * gain))), max(1, min(p.height, round(h * gain))))
            if p.keep_ratio
            else (p.width, p.height)
        )
        if p.backend == "pillow":
            from PIL import Image

            sample.data = np.asarray(
                Image.fromarray(sample.data).resize((nw, nh), Image.Resampling.BILINEAR)
            )
        else:
            sample.data = cv2.resize(sample.data, (nw, nh), interpolation=cv2.INTER_LINEAR)
        g = sample.geometry
        sx, sy = nw / w, nh / h
        sample.geometry = replace(
            g,
            scale=(g.scale[0] * sx, g.scale[1] * sy),
            padding=(g.padding[0] * sx, g.padding[1] * sy),
            gain=g.gain * gain if p.keep_ratio and g.gain is not None else None,
        )

    return image_step(plan, run)


class PaddingOptions(Section):
    width: int = Field(gt=0, strict=True)
    height: int = Field(gt=0, strict=True)
    value: int = Field(default=114, ge=0, le=255, strict=True)
    alignment: Literal["center", "top_left"] = "center"


@StepRegistry.register("padding")
def padding_step(config, **options):
    p = PaddingOptions.model_validate(options)

    def plan(state):
        require_hwc(state)
        if state.width is not None and (state.width > p.width or state.height > p.height):
            raise ValueError("padding target must not crop its input")
        state.height, state.width, state.transformed = p.height, p.width, True
        state.padded = True

    def run(sample):
        h, w = sample.data.shape[:2]
        if h > p.height or w > p.width:
            raise ValueError("padding target must not crop its input")
        left, top = ((p.width - w) // 2, (p.height - h) // 2) if p.alignment == "center" else (0, 0)
        canvas = np.full((p.height, p.width, 3), p.value, np.uint8)
        canvas[top : top + h, left : left + w] = sample.data
        sample.data = canvas
        g = sample.geometry
        sample.geometry = replace(g, padding=(g.padding[0] + left, g.padding[1] + top))

    return image_step(plan, run)


class LetterboxOptions(Section):
    width: int = Field(gt=0, strict=True)
    height: int = Field(gt=0, strict=True)
    value: int = Field(default=114, ge=0, le=255, strict=True)
    scale_up: bool = True
    rounding: Literal["floor", "ultralytics"] = "ultralytics"


@StepRegistry.register("letterbox")
def letterbox_step(config, **options):
    p = LetterboxOptions.model_validate(options)

    def plan(state):
        require_hwc(state)
        if state.transformed:
            raise ValueError(
                "letterbox must be the first geometry operation; use resize/padding for composition"
            )
        state.height, state.width, state.transformed = p.height, p.width, True
        state.padded = True

    def run(sample):
        sample.data, transform = letterbox(
            sample.data, p.height, p.width, p.value, scale_up=p.scale_up, rounding=p.rounding
        )
        sample.geometry = replace(
            sample.geometry, scale=transform.scale, gain=transform.gain, padding=transform.padding
        )

    return image_step(plan, run)


class ColorOptions(Section):
    source: Literal["BGR", "RGB"]
    target: Literal["BGR", "RGB"]


@StepRegistry.register("color")
def color_step(config, **options):
    p = ColorOptions.model_validate(options)

    def plan(state):
        if state.layout != "HWC" or state.color != p.source:
            raise ValueError("color step source/layout does not match preceding steps")
        state.color = p.target

    def run(sample):
        if sample.color != p.source or sample.layout != "HWC":
            raise ValueError("color step received unexpected source/layout")
        if p.source != p.target:
            sample.data = sample.data[..., ::-1]
        sample.color = p.target

    return image_step(plan, run)


class NormalizeOptions(Section):
    dtype: Literal["float32", "float16"] = "float32"
    scale: float | None = None
    divisor: float | None = None
    mean: tuple[float, float, float] = (0.0, 0.0, 0.0)
    std: tuple[float, float, float] = (1.0, 1.0, 1.0)

    @model_validator(mode="after")
    def check(self):
        if self.scale is not None and self.divisor is not None:
            raise ValueError("normalize accepts scale or divisor, not both")
        if (
            not all(math.isfinite(x) for x in (*self.mean, *self.std))
            or any(x <= 0 for x in self.std)
            or self.scale is not None
            and not math.isfinite(self.scale)
            or self.divisor is not None
            and (not math.isfinite(self.divisor) or self.divisor <= 0)
        ):
            raise ValueError("normalization must be finite with positive std/divisor")
        return self


@StepRegistry.register("normalize")
def normalize_step(config, **options):
    p = NormalizeOptions.model_validate(options)
    dtype = np.dtype(p.dtype)
    divisor = 1.0 if p.divisor is None else p.divisor
    mean = np.asarray([v * divisor for v in p.mean], dtype=dtype)
    std = np.asarray([v * divisor for v in p.std], dtype=dtype)
    if p.scale is not None and not np.isfinite(dtype.type(p.scale)):
        raise ValueError("scale exceeds selected dtype")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError("normalization parameters exceed selected dtype")

    def plan(state):
        if state.layout != "HWC":
            raise ValueError("normalize requires HWC data; put layout afterwards")
        state.dtype = p.dtype

    def run(sample):
        data = sample.data.astype(dtype)
        if p.scale is not None:
            sample.data = (data * dtype.type(p.scale) - mean) / std
        else:
            sample.data = (data - mean) / std
        if not np.isfinite(sample.data).all():
            raise ValueError("normalize generated nonfinite pixels")

    return image_step(plan, run)


class CastOptions(Section):
    dtype: Literal["float32", "float16", "uint8"]


@StepRegistry.register("cast")
def cast_step(config, **options):
    p = CastOptions.model_validate(options)

    def plan(state):
        state.dtype = p.dtype

    def run(sample):
        sample.data = sample.data.astype(p.dtype)

    return image_step(plan, run)


class LayoutOptions(Section):
    source: Literal["HWC", "CHW"]
    target: Literal["HWC", "CHW"]


@StepRegistry.register("layout")
def layout_step(config, **options):
    p = LayoutOptions.model_validate(options)

    def plan(state):
        if state.layout != p.source:
            raise ValueError("layout source does not match preceding steps")
        state.layout = p.target

    def run(sample):
        if p.source != p.target:
            sample.data = sample.data.transpose((2, 0, 1) if p.target == "CHW" else (1, 2, 0))
        sample.layout = p.target

    return image_step(plan, run)


class FaceAlignOptions(Section):
    aligned_input: bool = False


@StepRegistry.register("face_align")
def face_align_step(config, **options):
    p = FaceAlignOptions.model_validate(options)

    def plan(state):
        require_hwc(state)
        if state.transformed:
            raise ValueError(
                "face_align must precede geometry steps; landmarks use original coordinates"
            )
        state.height, state.width, state.transformed, state.restorable = 112, 112, True, False

    def run(sample):
        if sample.keypoints is not None:
            sample.data = align_face(sample.data, sample.keypoints)
            sample.keypoints = None
        elif not p.aligned_input:
            raise TypeError(
                "face_align requires FaceInput; aligned_input=true permits aligned crops"
            )
        elif sample.data.shape[:2] != (112, 112):
            raise ValueError("already aligned SFace crop must be 112x112")
        sample.geometry = replace(sample.geometry, restorable=False)

    return image_step(plan, run)


class PackOptions(Section):
    input_name: str
    sizes_name: str | None = None


@StepRegistry.register("pack")
def pack_step(config, **options):
    p = PackOptions.model_validate(options)
    specs = {s.name: s for s in config.inputs}
    expected = {p.input_name} | ({p.sizes_name} if p.sizes_name is not None else set())
    if set(specs) != expected or p.sizes_name == p.input_name:
        raise ValueError("pack tensor names must exactly match model inputs")
    image_spec = specs[p.input_name]
    if len(image_spec.shape) != 4:
        raise ValueError("pack image tensor must have rank 4")
    if p.sizes_name is not None:
        spec = specs[p.sizes_name]
        if spec.shape[1:] != (2,) or spec.dtype not in ("int64", "float32"):
            raise ValueError("target sizes must be int64/float32 [N,2]")
    final_shape = []

    def plan(state):
        shape = (
            (3, state.height, state.width)
            if state.layout == "CHW"
            else (state.height, state.width, 3)
        )
        if state.dtype != image_spec.dtype or any(
            w is not None and w != g for w, g in zip(image_spec.shape[1:], shape)
        ):
            raise ValueError(
                f"pack output layout/dtype/shape does not match model input {image_spec.name}"
            )
        final_shape.extend(shape)

    def run(context):
        samples = context.data
        if any(s.keypoints is not None for s in samples):
            raise ValueError("FaceInput landmarks must be consumed by face_align before pack")
        if samples:
            tensors = {p.input_name: np.ascontiguousarray(np.stack([s.data for s in samples]))}
        else:
            tensors = {
                p.input_name: np.empty((0, *(d or 0 for d in final_shape)), image_spec.dtype)
            }
        if p.sizes_name is not None:
            sizes = [(s.geometry.original_shape[1], s.geometry.original_shape[0]) for s in samples]
            tensors[p.sizes_name] = np.asarray(sizes, dtype=specs[p.sizes_name].dtype).reshape(
                -1, 2
            )
        context.data = PreparedInput(tensors, tuple(s.geometry for s in samples))
        context.kind = "prepared"

    return Step("images", "prepared", run, plan)
