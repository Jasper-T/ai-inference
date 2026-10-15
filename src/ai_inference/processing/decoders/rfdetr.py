"""RF-DETR reference: normalized cxcywh boxes + per-class sigmoid logits."""

import numpy as np
from pydantic import Field

from ai_inference.config import Section
from ai_inference.contracts.data import Prediction, RawOutput
from ai_inference.contracts.validation import validate_tensors
from ai_inference.processing.context import Geometry
from ai_inference.processing.geometry import clip_boxes, cxcywh_to_xyxy, restore_boxes


class Options(Section):
    boxes_name: str = "dets"
    logits_name: str = "labels"
    num_classes: int = Field(default=2, gt=0, strict=True)
    threshold: float = Field(default=0.4, ge=0, le=1, allow_inf_nan=False)
    clip_boxes: bool = False
    # Optional truncation for models whose output contract uses integer coordinates.
    integer_boxes: bool = False


class Decoder:
    def __init__(self, config, options: Options, width: int, height: int):
        self.config = config
        self.options = options
        self.width, self.height = width, height

    def __call__(self, outputs: RawOutput) -> list[Prediction]:
        prepared = outputs.prepared
        validate_tensors(outputs.tensors, self.config.outputs, prepared.count)
        boxes = outputs.tensors[self.options.boxes_name]
        logits = outputs.tensors[self.options.logits_name]
        if (
            boxes.shape[1] != logits.shape[1]
            or not np.isfinite(boxes).all()
            or not np.isfinite(logits).all()
        ):
            raise ValueError(
                "RF-DETR outputs contain inconsistent query counts or nonfinite values"
            )
        results = []
        for raw_boxes, raw_logits, meta in zip(boxes, logits, prepared.metadata):
            if not isinstance(meta, Geometry):
                raise ValueError("metadata does not belong to RF-DETR preprocessing")
            # Stable sigmoid, including very large positive/negative logits.
            z = np.exp(-np.abs(raw_logits))
            probability = np.where(raw_logits >= 0, 1 / (1 + z), z / (1 + z))
            ids = probability.argmax(axis=-1)
            scores = probability.max(axis=-1)
            keep = scores >= self.options.threshold
            b = raw_boxes[keep]
            xyxy = cxcywh_to_xyxy(b, clamp_wh=True)
            xyxy = xyxy.astype(np.float64) * [self.width, self.height, self.width, self.height]
            xyxy = restore_boxes(xyxy, meta.scale, meta.padding)
            if self.options.clip_boxes:
                xyxy = clip_boxes(xyxy, meta.original_shape)
            if self.options.integer_boxes:
                xyxy = np.trunc(xyxy)
            selected = ids[keep].astype(np.int64)
            results.append(
                Prediction(
                    xyxy.astype(np.float32),
                    scores[keep].astype(np.float32),
                    selected,
                    meta.original_shape,
                )
            )
        return results
