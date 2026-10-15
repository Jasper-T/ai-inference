"""Single model execution, input normalization, and response serialization."""

from collections.abc import Callable
from threading import Lock

import numpy as np

from ai_inference import FaceInput, FeaturePrediction, Model, Prediction
from ai_inference.processing.decoders.sface import alignment_matrix

from .inputs import InputError, decode_image
from .schemas import ImageMetadata, PredictionResponse, PredictResponse
from .settings import Settings


class Runtime:
    def __init__(self, settings: Settings, model_factory=Model):
        self.settings = settings
        self.model = model_factory(settings.model_config_path)
        declared_color = self.model.config.preprocess.get("source_color", "BGR")
        if declared_color != settings.input_color:
            raise ValueError("Service input_color must match model preprocess.source_color")
        self.requires_keypoints = any(
            "face_align" in step and not step["face_align"].get("aligned_input", False)
            for step in self.model.config.preprocess["steps"]
        )
        self.accepts_keypoints = any(
            "face_align" in step for step in self.model.config.preprocess["steps"]
        )
        self.lock = Lock()

    def predict(
        self, sources: list[Callable[[], bytes]], metadata: list[ImageMetadata]
    ) -> PredictResponse:
        if not sources or len(sources) > self.settings.max_batch_size:
            raise InputError("Invalid image batch size")
        if len(sources) != len(metadata):
            raise InputError("Metadata count must match image count")
        inputs, pixels = [], 0
        for index, (load, info) in enumerate(zip(sources, metadata)):
            try:
                if self.requires_keypoints and info.keypoints is None:
                    raise InputError("This model requires five face keypoints per image")
                if info.keypoints is not None:
                    if not self.accepts_keypoints:
                        raise InputError("This model does not accept face keypoints")
                    try:
                        with np.errstate(over="raise", invalid="raise", divide="raise"):
                            alignment_matrix(info.keypoints)
                    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                        raise InputError("Invalid or degenerate face keypoints") from None
                image = decode_image(load(), self.settings, self.settings.max_batch_pixels - pixels)
                if (
                    self.accepts_keypoints
                    and info.keypoints is None
                    and image.shape[:2] != (112, 112)
                ):
                    raise InputError("Already aligned face crops must be 112x112")
            except InputError as exc:
                exc.index = index
                raise
            pixels += image.shape[0] * image.shape[1]
            inputs.append(
                image if info.keypoints is None else FaceInput(image, np.array(info.keypoints))
            )
        results = self.model.predict(inputs)
        if len(results) != len(inputs):
            raise RuntimeError("Model result count does not match input count")
        serialized = []
        for result in results:
            fields = {"original_shape": result.original_shape}
            if isinstance(result, Prediction):
                fields.update(
                    type="detection",
                    boxes=result.boxes.tolist(),
                    scores=result.scores.tolist(),
                    class_ids=result.class_ids.tolist(),
                    keypoints=None if result.keypoints is None else result.keypoints.tolist(),
                    masks=None if result.masks is None else (result.masks != 0).tolist(),
                )
            elif isinstance(result, FeaturePrediction):
                fields.update(type="embedding", embedding=result.embedding.tolist())
            else:
                raise TypeError("Unsupported model result")
            serialized.append(PredictionResponse(**fields))
        return PredictResponse(results=serialized)
