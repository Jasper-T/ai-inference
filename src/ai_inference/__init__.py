"""Public SDK exports; implementations are grouped by lifecycle responsibility."""

from ai_inference.config import Config
from ai_inference.contracts.data import (
    FaceInput,
    FeaturePrediction,
    ModelInput,
    Prediction,
    PreparedInput,
    RawOutput,
    Result,
)
from ai_inference.contracts.processors import Postprocessor, Preprocessor
from ai_inference.engines import BaseEngine, Engine, OMEngine, ONNXRuntimeEngine
from ai_inference.engines.registry import EngineRegistry
from ai_inference.inference.batcher import BatchChunk, Batcher
from ai_inference.inference.predictor import DefaultPredictor, Predictor
from ai_inference.logging import LOG_FORMAT, log_context, log_event
from ai_inference.model import Model
from ai_inference.processing import Pipeline, State, Step, StepRegistry
from ai_inference.processing.context import Context, Geometry, SampleContext
from ai_inference.visualization.plotter import Plotter

__version__ = "0.2.0"
__all__ = [
    "Model",
    "Config",
    "PreparedInput",
    "RawOutput",
    "Prediction",
    "FaceInput",
    "FeaturePrediction",
    "Result",
    "ModelInput",
    "EngineRegistry",
    "Engine",
    "BaseEngine",
    "ONNXRuntimeEngine",
    "OMEngine",
    "Preprocessor",
    "Postprocessor",
    "Plotter",
    "Predictor",
    "DefaultPredictor",
    "Batcher",
    "BatchChunk",
    "Context",
    "Geometry",
    "SampleContext",
    "Pipeline",
    "Step",
    "StepRegistry",
    "State",
    "LOG_FORMAT",
    "log_context",
    "log_event",
]
