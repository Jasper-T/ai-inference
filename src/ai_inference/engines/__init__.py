"""Physical inference backends and their registration interface."""

from ai_inference.engines.base import BaseEngine, Engine
from ai_inference.engines.om import OMEngine
from ai_inference.engines.onnxruntime import ONNXRuntimeEngine
from ai_inference.engines.registry import EngineRegistry
from ai_inference.engines.tensorrt import TensorRTEngine

__all__ = [
    "Engine",
    "BaseEngine",
    "ONNXRuntimeEngine",
    "OMEngine",
    "TensorRTEngine",
    "EngineRegistry",
]
