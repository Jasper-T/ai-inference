"""Backend factory registration; no model loading or request execution."""

from ai_inference.contracts.registry import Registry

EngineRegistry = Registry("engine")
