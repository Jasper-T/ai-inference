"""Original RT-DETR export with deployed postprocessor, not RF-DETR logits."""

import numpy as np
from pydantic import Field

from ai_inference.config import Section
from ai_inference.contracts.data import Prediction
from ai_inference.contracts.validation import validate_tensors
from ai_inference.processing.context import Geometry
from ai_inference.processing.geometry import clip_boxes


class Options(Section):
    labels_name: str = "labels"
    boxes_name: str = "boxes"
    scores_name: str = "scores"
    num_classes: int = Field(gt=0, strict=True)
    threshold: float = Field(default=0.5, ge=0, le=1, allow_inf_nan=False)
    clip_boxes: bool = True


class Decoder:
    def __init__(self, config, options: Options):
        self.config = config
        self.options = options

    def __call__(self, outputs):
        post = self.options
        validate_tensors(outputs.tensors, self.config.outputs, outputs.prepared.count)
        tensors = outputs.tensors
        if any(not np.isfinite(v).all() for v in tensors.values()):
            raise ValueError("nonfinite RT-DETR output")
        results = []
        for labels, boxes, scores, meta in zip(
            tensors[post.labels_name],
            tensors[post.boxes_name],
            tensors[post.scores_name],
            outputs.prepared.metadata,
        ):
            if not isinstance(meta, Geometry):
                raise ValueError("invalid RT-DETR metadata")
            if meta.padding != (0.0, 0.0) or not meta.restorable:
                raise ValueError("RT-DETR deploy requires direct resize without padding")
            if labels.shape != scores.shape or boxes.shape != (*scores.shape, 4):
                raise ValueError("inconsistent RT-DETR output counts")
            if (
                ((labels < 0) | (labels >= post.num_classes)).any()
                or ((scores < 0) | (scores > 1)).any()
                or (boxes[:, 2:] < boxes[:, :2]).any()
            ):
                raise ValueError("invalid RT-DETR labels/scores/boxes")
            keep = scores >= post.threshold
            ids = labels[keep].astype(np.int64)
            b = boxes[keep]
            if post.clip_boxes:
                b = clip_boxes(b, meta.original_shape)
            # Export already scales boxes to original coordinates. No NMS/rescale.
            results.append(
                Prediction(
                    b.astype(np.float32), scores[keep].astype(np.float32), ids, meta.original_shape
                )
            )
        return results
