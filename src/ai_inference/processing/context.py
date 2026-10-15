"""Request-local processing data and geometry, independent of model families."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Geometry:
    original_shape: tuple[int, int]
    scale: tuple[float, float] = (1.0, 1.0)
    padding: tuple[float, float] = (0.0, 0.0)
    gain: float | None = 1.0
    restorable: bool = True


@dataclass
class SampleContext:
    data: np.ndarray
    geometry: Geometry
    color: str
    layout: str = "HWC"
    device: str = "cpu"
    keypoints: np.ndarray | None = None


@dataclass
class Context:
    """Each invocation owns its context; reusable steps must not hold request data."""

    data: Any
    kind: str
    device: str = "cpu"
    attributes: dict[str, Any] = field(default_factory=dict)
