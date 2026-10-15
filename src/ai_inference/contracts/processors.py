"""Backend-independent processing contracts; implementations may use any library.

Custom processor pairs may define their own per-image metadata. When replacing
only one processor, preserve the metadata contract of its counterpart. Tensor
values currently remain CPU NumPy arrays as required by the engine/batching API.
Processing placement is separate from runtime provider selection; Model rejects
unimplemented processing devices and per-step plans before engine creation.
"""

from typing import Protocol

from ai_inference.contracts.data import ModelInput, PreparedInput, RawOutput, Result


class Preprocessor(Protocol):
    def __call__(self, images: ModelInput, /) -> PreparedInput: ...


class Postprocessor(Protocol):
    def __call__(self, outputs: RawOutput, /) -> list[Result]: ...
