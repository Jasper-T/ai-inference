"""Reusable result operations, independent of model output encoding."""

from dataclasses import replace

import numpy as np
from pydantic import Field

from ai_inference.config import Section
from ai_inference.contracts.data import FeaturePrediction, Prediction
from ai_inference.processing.pipeline import Step, StepRegistry


class FilterOptions(Section):
    class_ids: tuple[int, ...] | None = None
    threshold: float = Field(default=0.0, ge=0, le=1, allow_inf_nan=False)


@StepRegistry.register("filter_detections")
def filter_detections(config, **options):
    p = FilterOptions.model_validate(options)

    def run(context):
        results = []
        for result in context.data:
            if not isinstance(result, Prediction):
                raise TypeError("filter_detections requires detection results")
            keep = result.scores >= p.threshold
            if p.class_ids is not None:
                keep &= np.isin(result.class_ids, p.class_ids)
            results.append(
                replace(
                    result,
                    boxes=result.boxes[keep],
                    scores=result.scores[keep],
                    class_ids=result.class_ids[keep],
                    keypoints=None if result.keypoints is None else result.keypoints[keep],
                    masks=None if result.masks is None else result.masks[keep],
                )
            )
        context.data = results

    return Step("results", "results", run, lambda state: None)


@StepRegistry.register("l2_normalize")
def l2_normalize(config, **options):
    Section.model_validate(options)

    def run(context):
        results = []
        for result in context.data:
            if not isinstance(result, FeaturePrediction):
                raise TypeError("l2_normalize requires feature results")
            feature = result.embedding
            norm = np.linalg.norm(feature.astype(np.float64))
            if not np.isfinite(feature).all() or not np.isfinite(norm) or norm <= 0:
                raise ValueError("L2 normalization requires finite nonzero features")
            results.append(replace(result, embedding=(feature / norm).astype(np.float32)))
        context.data = results

    return Step("results", "results", run, lambda state: None)
