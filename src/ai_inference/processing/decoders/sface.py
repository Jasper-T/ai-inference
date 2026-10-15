"""SFace 2021dec: five-point similarity alignment, RGB raw float pixels."""

import cv2
import numpy as np
from pydantic import Field

from ai_inference.config import Section
from ai_inference.contracts.data import FeaturePrediction
from ai_inference.contracts.validation import validate_tensors
from ai_inference.processing.common import image_list
from ai_inference.processing.context import Geometry

TARGET_POINTS = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float64,
)


class Options(Section):
    output_name: str = "fc1"
    dimensions: int = Field(default=128, gt=0, strict=True)


def alignment_matrix(keypoints):
    """Validate five landmarks and return the similarity transform without warping."""
    source = np.asarray(keypoints, dtype=np.float64)
    if source.shape != (5, 2) or not np.isfinite(source).all():
        raise ValueError("SFace requires five finite xy landmarks in YuNet order")
    x = source - source.mean(0)
    y = TARGET_POINTS - TARGET_POINTS.mean(0)
    variance = np.sum(x * x) / 5
    if variance < 1e-6 or np.linalg.matrix_rank(x) < 2:
        raise ValueError("degenerate face landmarks")
    u, s, vt = np.linalg.svd(y.T @ x / 5)
    sign = np.array([1.0, 1.0 if np.linalg.det(u @ vt) >= 0 else -1.0])
    rotation = u @ np.diag(sign) @ vt
    scale = np.sum(s * sign) / variance
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("invalid face alignment scale")
    return np.column_stack(
        (scale * rotation, TARGET_POINTS.mean(0) - scale * rotation @ source.mean(0))
    )


def align_face(image, keypoints):
    image_list(image)
    return cv2.warpAffine(
        image,
        alignment_matrix(keypoints),
        (112, 112),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )


class Decoder:
    def __init__(self, config, options: Options):
        self.config = config
        self.options = options

    def __call__(self, outputs):
        validate_tensors(outputs.tensors, self.config.outputs, outputs.prepared.count)
        features = outputs.tensors[self.options.output_name]
        if not np.isfinite(features).all():
            raise ValueError("nonfinite SFace feature")
        norms = np.linalg.norm(features.astype(np.float64), axis=1)
        if (norms <= 0).any():
            raise ValueError("zero SFace feature")
        if any(not isinstance(meta, Geometry) for meta in outputs.prepared.metadata):
            raise ValueError("SFace requires Geometry metadata")
        return [
            FeaturePrediction(v.copy(), meta.original_shape)
            for v, meta in zip(features, outputs.prepared.metadata)
        ]
