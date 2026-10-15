"""Transport-free intermediate and prediction types; no business metadata."""

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class PreparedInput:
    """Unpadded tensors [N,...], one processing metadata entry per real sample."""

    tensors: Mapping[str, np.ndarray]
    metadata: tuple[Any, ...]

    @property
    def count(self) -> int:
        return len(self.metadata)


@dataclass(frozen=True)
class RawOutput:
    """Unpadded, ordered tensors [N,...]; metadata travels with its own request."""

    tensors: Mapping[str, np.ndarray]
    prepared: PreparedInput


@dataclass(frozen=True)
class Prediction:
    """One image: xyxy pixels, float32 scores, int64 class IDs (decoder-defined order)."""

    boxes: np.ndarray
    scores: np.ndarray
    class_ids: np.ndarray
    original_shape: tuple[int, int]

    # Optional instance data in original image coordinates; detection-only callers
    # retain the same four constructor arguments and existing fields.
    keypoints: np.ndarray | None = None  # [instances, points, 2], xy pixels
    masks: np.ndarray | None = None  # [instances, H, W], original-resolution masks


@dataclass(frozen=True)
class FeaturePrediction:
    """One sample's finite float32 feature vector (L2 normalized by default)."""

    embedding: np.ndarray
    original_shape: tuple[int, int]


@dataclass(frozen=True)
class FaceInput:
    """Full uint8 image and five original-coordinate points in YuNet order:
    right eye, left eye, nose, right mouth, left mouth.
    """

    image: np.ndarray
    keypoints: np.ndarray


Result = Prediction | FeaturePrediction
ModelInput = np.ndarray | list[np.ndarray] | FaceInput | list[FaceInput]
