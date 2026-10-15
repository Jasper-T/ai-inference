"""Ultralytics YOLOv8/YOLO11 detection exports with nms=False."""

import math
from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from ai_inference.config import Config, Section
from ai_inference.contracts.data import Prediction, RawOutput
from ai_inference.contracts.validation import validate_tensors
from ai_inference.processing.context import Geometry
from ai_inference.processing.geometry import clip_boxes, cxcywh_to_xyxy, nms, restore_boxes


class Options(Section):
    output_name: str | None = None
    layout: Literal["NCQ", "NQC"] = "NCQ"
    num_classes: int = Field(gt=0, strict=True)
    confidence_threshold: float = Field(default=0.25, ge=0, le=1)
    iou_threshold: float = Field(default=0.7, ge=0, le=1)
    max_detections: int = Field(default=300, gt=0, strict=True)
    max_nms: int = Field(default=30000, gt=0, strict=True)
    agnostic_nms: bool = False
    clip_boxes: bool = True

    def resolve_output(self, config: Config) -> "Options":
        outputs = config.outputs
        if outputs is None or len(outputs) != 1:
            raise ValueError("YOLO detection requires exactly one output")
        name = outputs[0].name
        if self.output_name is not None and self.output_name != name:
            raise ValueError("output_name must match outputs[0].name")
        return (
            self if self.output_name is not None else self.model_copy(update={"output_name": name})
        )

    @model_validator(mode="after")
    def check_classes(self):
        if not all(math.isfinite(v) for v in (self.confidence_threshold, self.iou_threshold)):
            raise ValueError("thresholds must be finite")
        return self


class Decoder:
    def __init__(self, config, options: Options):
        self.config = config
        self.options = options

    def __call__(self, outputs: RawOutput) -> list[Prediction]:
        validate_tensors(outputs.tensors, self.config.outputs, outputs.prepared.count)
        name = self.options.output_name
        if name is None:
            raise ValueError("YOLO output_name must be resolved before decoding")
        values = outputs.tensors[name].astype(np.float32)
        if not np.isfinite(values).all():
            raise ValueError("YOLO output contains nonfinite values")
        if self.options.layout == "NCQ":
            values = values.transpose(0, 2, 1)
        results = []
        p = self.options
        for rows, meta in zip(values, outputs.prepared.metadata):
            if not isinstance(meta, Geometry):
                raise ValueError("metadata does not belong to YOLO preprocessing")
            probabilities = rows[:, 4:]
            if np.any((probabilities < 0) | (probabilities > 1)):
                raise ValueError("YOLO expects class probabilities, not logits or objectness")
            ids = probabilities.argmax(axis=1)
            scores = probabilities.max(axis=1)
            mask = (scores >= p.confidence_threshold) & (rows[:, 2:4] > 0).all(axis=1)
            selected = np.flatnonzero(mask)
            selected = selected[np.argsort(-scores[selected], kind="stable")[: p.max_nms]]
            xywh = rows[selected, :4].astype(np.float64)
            boxes = cxcywh_to_xyxy(xywh)
            scores, ids = scores[selected], ids[selected]
            keep = nms(boxes, scores, ids, p.iou_threshold, p.max_detections, p.agnostic_nms)
            boxes, scores, ids = boxes[keep], scores[keep], ids[keep]
            boxes = restore_boxes(boxes, meta.gain, meta.padding)
            if p.clip_boxes:
                boxes = clip_boxes(boxes, meta.original_shape)
            results.append(
                Prediction(
                    boxes.astype(np.float32), scores, ids.astype(np.int64), meta.original_shape
                )
            )
        return results
