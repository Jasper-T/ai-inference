"""Backend-independent batch preparation and restoration; no engine execution."""

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass

import numpy as np

from ai_inference.config import Config
from ai_inference.contracts.data import PreparedInput, RawOutput
from ai_inference.contracts.validation import validate_tensors
from ai_inference.logging import log_event


@dataclass(frozen=True)
class BatchChunk:
    tensors: Mapping[str, np.ndarray]
    start: int
    valid_count: int
    physical_count: int
    index: int
    total: int

    def __post_init__(self):
        if (
            self.start < 0
            or not 0 < self.valid_count <= self.physical_count
            or not 1 <= self.index <= self.total
        ):
            raise ValueError("invalid batch chunk bounds or counts")

    @property
    def padded_count(self) -> int:
        return self.physical_count - self.valid_count


class Batcher:
    """Share immutable policy; keep chunks and collected outputs request-local."""

    def __init__(self, config: Config):
        if not config.has_tensor_contract:
            raise ValueError("Batcher requires a resolved tensor contract")
        self.config = config

    def split(self, prepared: PreparedInput) -> Iterator[BatchChunk]:
        """Validate logical inputs, then yield contiguous, optionally padded feeds."""
        validate_tensors(prepared.tensors, self.config.inputs, prepared.count)
        policy = self.config.runtime.batch
        total = (prepared.count + policy.size - 1) // policy.size
        padded = total * policy.size - prepared.count if policy.mode == "fixed" else 0
        log_event(
            "infer",
            "batching",
            f"images={prepared.count} | mode={policy.mode} | batches={total} | size={policy.size} | padding={padded}",
            model=self.config.model.name,
            sample_count=prepared.count,
            batch_count=total,
            batch_mode=policy.mode,
            batch_size=policy.size,
            padded_samples=padded,
        )
        for start in range(0, prepared.count, policy.size):
            valid = min(policy.size, prepared.count - start)
            physical = policy.size if policy.mode == "fixed" else valid
            feeds = {}
            for name, tensor in prepared.tensors.items():
                chunk = tensor[start : start + valid]
                if physical > valid:
                    pad = (
                        np.repeat(chunk[-1:], physical - valid, axis=0)
                        if policy.padding == "repeat_last"
                        else np.zeros((physical - valid, *chunk.shape[1:]), dtype=chunk.dtype)
                    )
                    chunk = np.concatenate((chunk, pad), axis=0)
                feeds[name] = np.ascontiguousarray(chunk)
            yield BatchChunk(feeds, start, valid, physical, start // policy.size + 1, total)

    def trim(self, outputs: Mapping[str, np.ndarray], chunk: BatchChunk) -> dict[str, np.ndarray]:
        """Validate physical outputs and detach valid samples from backend buffers."""
        validate_tensors(outputs, self.config.outputs, chunk.physical_count)
        specs = self.config.outputs
        if specs is None:
            raise ValueError("Batcher requires a resolved output tensor contract")
        return {spec.name: outputs[spec.name][: chunk.valid_count].copy() for spec in specs}

    def merge(
        self, parts: Iterable[Mapping[str, np.ndarray]], prepared: PreparedInput
    ) -> RawOutput:
        """Merge ordered, already trimmed parts and preserve request metadata."""
        pieces = list(parts)
        specs = self.config.outputs
        if specs is None:
            raise ValueError("Batcher requires a resolved output tensor contract")
        if prepared.count == 0 and not pieces:
            # Unknown non-batch dimensions cannot be inferred without execution.
            tensors = {}
            for spec in specs:
                shape, dtype = spec.shape, spec.dtype
                if shape is None or dtype is None:
                    raise ValueError("Batcher requires a resolved output tensor contract")
                tensors[spec.name] = np.empty((0, *(d or 0 for d in shape[1:])), dtype=dtype)
        else:
            if not pieces:
                raise ValueError("nonempty request requires output batches")
            for part in pieces:
                if not isinstance(part, Mapping) or not part:
                    raise ValueError("output batches must be named tensor mappings")
                first = next(iter(part.values()))
                if not isinstance(first, np.ndarray) or first.ndim == 0:
                    raise ValueError("output batches require a batch dimension")
                validate_tensors(part, self.config.outputs, first.shape[0])
            tensors = {
                s.name: np.concatenate([part[s.name] for part in pieces], axis=0) for s in specs
            }
        validate_tensors(tensors, self.config.outputs, prepared.count)
        return RawOutput(tensors, prepared)
