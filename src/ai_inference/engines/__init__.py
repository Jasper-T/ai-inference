"""Physical inference backends and their registration interface."""

from ai_inference.engines.base import BaseEngine, Engine
from ai_inference.engines.om import OMEngine
from ai_inference.engines.onnxruntime import ONNXRuntimeEngine
from ai_inference.engines.registry import EngineRegistry

__all__ = ["Engine", "BaseEngine", "ONNXRuntimeEngine", "OMEngine", "EngineRegistry"]
