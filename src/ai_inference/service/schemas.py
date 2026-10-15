"""HTTP contracts shared by all input sources."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

Point = tuple[float, float]
Keypoints = Annotated[list[Point], Field(min_length=5, max_length=5)]


class ImageMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    keypoints: Keypoints | None = None


class Base64Image(ImageMetadata):
    data: str = Field(min_length=1)


class URLImage(ImageMetadata):
    url: str = Field(min_length=1, max_length=4096)


class Base64Request(BaseModel):
    model_config = ConfigDict(extra="forbid")
    images: list[Base64Image] = Field(min_length=1)


class URLRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    images: list[URLImage] = Field(min_length=1)


class PredictionResponse(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    type: Literal["detection", "embedding"]
    original_shape: tuple[int, int]
    boxes: list[list[float]] | None = None
    scores: list[float] | None = None
    class_ids: list[int] | None = None
    keypoints: list[list[list[float]]] | None = None
    masks: list[list[list[bool]]] | None = None
    embedding: list[float] | None = None


class PredictResponse(BaseModel):
    results: list[PredictionResponse]
