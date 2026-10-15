"""Shared CPU image validation and letterbox operation."""

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np


def image_list(images: np.ndarray | list[np.ndarray]) -> list[np.ndarray]:
    if isinstance(images, np.ndarray):
        images = [images]
    if not isinstance(images, list):
        raise TypeError("images must be one HWC uint8 ndarray or a list of them")
    for image in images:
        if (
            not isinstance(image, np.ndarray)
            or image.dtype != np.uint8
            or image.ndim != 3
            or image.shape[2] != 3
            or min(image.shape[:2]) <= 0
        ):
            raise ValueError("each image must be nonempty HWC uint8 with 3 channels")
    return images


@dataclass(frozen=True)
class ImageTransform:
    """Actual x/y scales and ideal uniform gain are deliberately distinct."""

    original_shape: tuple[int, int]
    scale: tuple[float, float]
    gain: float | None
    padding: tuple[int, int]


def letterbox(
    image: np.ndarray,
    height: int,
    width: int,
    pad_value: int = 114,
    *,
    scale_up: bool = True,
    rounding: Literal["floor", "ultralytics"] = "floor",
) -> tuple[np.ndarray, ImageTransform]:
    h, w = image.shape[:2]
    gain = min(width / w, height / h)
    if not scale_up:
        gain = min(gain, 1.0)
    nw, nh = max(1, min(width, round(w * gain))), max(1, min(height, round(h * gain)))
    if rounding == "floor":
        left, top = (width - nw) // 2, (height - nh) // 2
    elif rounding == "ultralytics":
        left, top = round((width - nw) / 2 - 0.1), round((height - nh) / 2 - 0.1)
    else:
        raise ValueError("unknown letterbox rounding policy")
    canvas = np.full((height, width, 3), pad_value, np.uint8)
    canvas[top : top + nh, left : left + nw] = cv2.resize(
        image, (nw, nh), interpolation=cv2.INTER_LINEAR
    )
    return canvas, ImageTransform((h, w), (nw / w, nh / h), gain, (left, top))
