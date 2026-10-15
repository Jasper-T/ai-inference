"""ONNX Runtime execution only: no image, detection or transport logic."""

from collections.abc import Mapping

import numpy as np
from pydantic import Field, field_validator, model_validator

from ai_inference.config import Config, Section
from ai_inference.engines.base import BaseEngine, TensorMetadata
from ai_inference.engines.registry import EngineRegistry


class ProviderConfig(Section):
    name: str = Field(min_length=1)
    options: dict[str, str] = Field(default_factory=dict)


PROVIDER_ALIASES = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "trt": "TensorrtExecutionProvider",
}


class ONNXRuntimeOptions(Section):
    providers: tuple[ProviderConfig, ...] = Field(
        default_factory=lambda: (
            ProviderConfig(name="CUDAExecutionProvider", options={"device_id": "0"}),
        ),
        min_length=1,
    )
    intra_op_num_threads: int = Field(default=1, ge=0, strict=True)

    @field_validator("providers", mode="before")
    @classmethod
    def expand_providers(cls, value):
        items = [value] if isinstance(value, str) else value
        if not isinstance(items, (list, tuple)):
            raise ValueError("providers must be a name or ordered list")
        expanded = []
        for item in items:
            if isinstance(item, str):
                name = PROVIDER_ALIASES.get(item, item)
                if name not in PROVIDER_ALIASES.values():
                    raise ValueError(f"unknown provider alias: {item}")
                options = {"device_id": "0"} if name != "CPUExecutionProvider" else {}
                expanded.append({"name": name, "options": options})
            else:
                # Object form configures advanced provider options.
                expanded.append(item)
        return expanded

    @model_validator(mode="after")
    def unique_providers(self):
        names = [provider.name for provider in self.providers]
        if len(names) != len(set(names)):
            raise ValueError("duplicate provider names")
        return self


@EngineRegistry.register("onnxruntime")
class ONNXRuntimeEngine(BaseEngine):
    def __init__(self, config: Config):
        super().__init__(config)
        runtime = ONNXRuntimeOptions.model_validate(config.runtime.options)
        self._load_session(runtime)
        # A broken model contract is not a provider failure; don't retry it.
        self._load_tensor_contract()

    def _load_session(self, runtime: ONNXRuntimeOptions) -> None:
        """Create the session using only explicitly configured providers."""
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "Install a compatible ONNX Runtime distribution; see README (GPU/PPU)."
            ) from exc
        requested = tuple(p.name for p in runtime.providers)
        available = set(ort.get_available_providers())
        candidates = [p for p in runtime.providers if p.name in available]
        if not candidates:
            raise RuntimeError(f"Requested ONNX Runtime providers unavailable: {list(requested)}")
        self.requested_providers = requested
        # A single candidate is strict. Longer lists allow initialization fallback
        # to the next explicitly configured provider, never to an implicit one.
        for start in range(len(candidates)):
            chosen = candidates[start:]
            names = [p.name for p in chosen]
            options = ort.SessionOptions()
            options.intra_op_num_threads = runtime.intra_op_num_threads
            if "CPUExecutionProvider" not in names:
                options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
            providers = [(p.name, dict(p.options)) for p in chosen]
            try:
                session = ort.InferenceSession(
                    str(self.config.model.path), sess_options=options, providers=providers
                )
                session.disable_fallback()  # No automatic retry after execution failure.
                active = tuple(session.get_providers())
                # CPU can be registered by ORT even when CPU graph fallback is disabled.
                unexpected = set(active) - set(names) - {"CPUExecutionProvider"}
                if chosen[0].name not in active or unexpected:
                    raise RuntimeError(
                        f"ORT provider initialization failed: requested {names}, active {active}"
                    )
            except Exception:
                if start + 1 == len(candidates):
                    raise
                continue
            self.session = session
            self.providers = active
            self.selected_providers = tuple(name for name in names if name in active)
            return

    def read_tensor_metadata(self) -> dict[str, tuple[TensorMetadata, ...]]:
        """Read actual session names, dtypes and shapes, keeping symbolic axes."""
        types = {
            "float32": "tensor(float)",
            "float16": "tensor(float16)",
            "float64": "tensor(double)",
            "int64": "tensor(int64)",
            "int32": "tensor(int32)",
            "uint8": "tensor(uint8)",
            "bool": "tensor(bool)",
        }
        dtypes = {value: key for key, value in types.items()}
        metadata = {}
        for kind, nodes in (
            ("inputs", self.session.get_inputs()),
            ("outputs", self.session.get_outputs()),
        ):
            tensors = []
            for node in nodes:
                if node.type not in dtypes:
                    raise ValueError(f"unsupported ONNX {kind} dtype: {node.name}: {node.type}")
                tensors.append(TensorMetadata(node.name, dtypes[node.type], tuple(node.shape)))
            metadata[kind] = tuple(tensors)
        return metadata

    def _infer(self, tensors: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        outputs = self.config.outputs
        if outputs is None:
            raise ValueError("ONNX output tensor contract must be resolved before inference")
        names = [s.name for s in outputs]
        values = self.session.run(
            names, {name: np.ascontiguousarray(value) for name, value in tensors.items()}
        )
        result = {}
        for name, value in zip(names, values):
            if not isinstance(value, np.ndarray):
                raise ValueError("ONNX outputs must be ndarrays")
            result[name] = value
        return result
