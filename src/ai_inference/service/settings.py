"""Service-only settings; relative configuration paths use the working directory."""

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_config_path: Path
    input_color: Literal["BGR", "RGB"] = "BGR"
    max_request_bytes: int = Field(default=24 * 1024 * 1024, gt=0)
    max_image_bytes: int = Field(default=8 * 1024 * 1024, gt=0)
    max_image_pixels: int = Field(default=16_000_000, gt=0)
    max_batch_pixels: int = Field(default=32_000_000, gt=0)
    max_batch_size: int = Field(default=8, gt=0)
    url_timeout_seconds: float = Field(default=10, gt=0, le=120)

    @classmethod
    def from_env(cls):
        fields = {
            key: os.environ[f"AI_INFERENCE_{key.upper()}"]
            for key in cls.model_fields
            if f"AI_INFERENCE_{key.upper()}" in os.environ
        }
        # Keep the documented deployment variable concise.
        if "AI_INFERENCE_MODEL_CONFIG" in os.environ:
            fields["model_config_path"] = os.environ["AI_INFERENCE_MODEL_CONFIG"]
        return cls.model_validate(fields)
