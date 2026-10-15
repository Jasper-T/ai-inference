"""Shared tensor contracts for logical requests and physical engine batches."""

from collections.abc import Mapping

import numpy as np

from ai_inference.config import Config, TensorSpec


def validate_tensors(
    tensors: Mapping[str, np.ndarray], specs: tuple[TensorSpec, ...] | None, count: int
) -> None:
    if not isinstance(tensors, Mapping):
        raise ValueError("tensors must be a named tensor mapping")
    if specs is None or any(s.dtype is None or s.shape is None for s in specs):
        raise ValueError("tensor contract must be resolved before validation")
    if set(tensors) != {s.name for s in specs}:
        raise ValueError("tensor names differ from configured contract")
    for spec in specs:
        shape = spec.shape
        dtype = spec.dtype
        if shape is None or dtype is None:
            raise ValueError("tensor contract must be resolved before validation")
        tensor = tensors[spec.name]
        if not isinstance(tensor, np.ndarray) or tensor.dtype != np.dtype(dtype):
            raise ValueError(f"{spec.name}: expected ndarray dtype {dtype}")
        if tensor.ndim != len(shape) or tensor.shape[0] != count:
            raise ValueError(
                f"{spec.name}: invalid rank or batch dimension {tensor.shape}; expected count={count}, rank={len(shape)}"
            )
        if any(want is not None and got != want for got, want in zip(tensor.shape[1:], shape[1:])):
            raise ValueError(f"{spec.name}: unexpected shape {tensor.shape}; spec {spec.shape}")


def validate_physical_inputs(tensors: Mapping[str, np.ndarray], config: Config) -> int:
    """Check physical batch size in addition to names, dtype and dimensions."""
    if not isinstance(tensors, Mapping) or not tensors:
        raise ValueError("inputs must be a nonempty tensor mapping")
    first = next(iter(tensors.values()))
    if not isinstance(first, np.ndarray) or first.ndim == 0:
        raise ValueError("input tensors must be ndarrays with a batch dimension")
    count = first.shape[0]
    policy = config.runtime.batch
    if count <= 0 or count > policy.size or (policy.mode == "fixed" and count != policy.size):
        raise ValueError("invalid physical batch size")
    validate_tensors(tensors, config.inputs, count)
    return count
