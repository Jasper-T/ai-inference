"""YuNet 2023mar float model's 12-output stride contract."""

from typing import Literal

import numpy as np
from pydantic import Field

from ai_inference.config import Section
from ai_inference.contracts.data import Prediction
from ai_inference.contracts.validation import validate_tensors
from ai_inference.processing.context import Geometry
from ai_inference.processing.geometry import clip_boxes, nms

STRIDES = (8, 16, 32)
CHANNELS = {"cls": 1, "obj": 1, "bbox": 4, "kps": 10}


class Options(Section):
    num_classes: Literal[1] = 1
    threshold: float = Field(default=0.8, ge=0, le=1, allow_inf_nan=False)
    nms_threshold: float = Field(default=0.3, ge=0, le=1, allow_inf_nan=False)
    top_k: int = Field(default=5000, gt=0, strict=True)
    max_detections: int = Field(default=100, gt=0, strict=True)
    clip_boxes: bool = True


class Decoder:
    def __init__(self, config, options: Options, width: int, height: int):
        self.config = config
        self.options = options
        self.width, self.height = width, height

    def __call__(self, outputs):
        validate_tensors(outputs.tensors, self.config.outputs, outputs.prepared.count)
        if any(not np.isfinite(v).all() for v in outputs.tensors.values()):
            raise ValueError("nonfinite YuNet output")
        results = []
        for sample, meta in enumerate(outputs.prepared.metadata):
            if not isinstance(meta, Geometry):
                raise ValueError("invalid YuNet metadata")
            boxes, points, scores = [], [], []
            for stride in STRIDES:
                t = {kind: outputs.tensors[f"{kind}_{stride}"][sample] for kind in CHANNELS}
                score = np.sqrt(np.clip(t["cls"][:, 0], 0, 1) * np.clip(t["obj"][:, 0], 0, 1))
                indices = np.flatnonzero(score >= self.options.threshold)
                cell = np.column_stack(
                    (indices % (self.width // stride), indices // (self.width // stride))
                )
                b = t["bbox"][indices].astype(np.float64)
                center = (cell + b[:, :2]) * stride
                with np.errstate(over="ignore"):
                    wh = np.exp(b[:, 2:]) * stride
                decoded = np.concatenate((center - wh / 2, center + wh / 2), axis=1)
                if not np.isfinite(decoded).all():
                    raise ValueError("YuNet box decode overflow")
                boxes.append(
                    (decoded - [*meta.padding, *meta.padding]) / [*meta.scale, *meta.scale]
                )
                points.append(
                    ((t["kps"][indices].reshape(-1, 5, 2) + cell[:, None]) * stride - meta.padding)
                    / meta.scale
                )
                scores.append(score[indices])
            b, k, s = np.concatenate(boxes), np.concatenate(points), np.concatenate(scores)
            order = np.argsort(-s, kind="stable")[: self.options.top_k]
            keep = order[
                nms(
                    b[order],
                    s[order],
                    np.zeros(len(order), np.int64),
                    self.options.nms_threshold,
                    self.options.max_detections,
                    True,
                )
            ]
            selected = b[keep]
            if self.options.clip_boxes:
                selected = clip_boxes(selected, meta.original_shape)
            limit = np.finfo(np.float32).max
            if (np.abs(selected) > limit).any() or (np.abs(k[keep]) > limit).any():
                raise ValueError("YuNet restored coordinates exceed float32 range")
            results.append(
                Prediction(
                    selected.astype(np.float32),
                    s[keep].astype(np.float32),
                    np.zeros(len(keep), np.int64),
                    meta.original_shape,
                    keypoints=k[keep].astype(np.float32),
                )
            )
        return results
