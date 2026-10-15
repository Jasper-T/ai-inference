"""Static Ascend OM execution through ais_bench, without image processing."""

from collections.abc import Mapping

import numpy as np
from pydantic import Field

from ai_inference.config import Config, Section
from ai_inference.engines.base import BaseEngine, TensorMetadata
from ai_inference.engines.registry import EngineRegistry


class OMOptions(Section):
    device_id: int = Field(default=0, ge=0, strict=True)


@EngineRegistry.register("om")
class OMEngine(BaseEngine):
    def __init__(self, config: Config):
        super().__init__(config)
        options = OMOptions.model_validate(config.runtime.options)
        if config.runtime.batch.mode != "fixed":
            raise ValueError("OMEngine currently supports static fixed-batch inference only")
        self._load_session(options)
        self._load_tensor_contract()

    def _load_session(self, options: OMOptions) -> None:
        try:
            from ais_bench.infer.interface import InferSession
        except ImportError as exc:
            raise ImportError(
                "OMEngine requires compatible ais_bench and ACLRuntime in an Ascend/CANN environment"
            ) from exc
        self.session = InferSession(options.device_id, str(self.config.model.path))

    def read_tensor_metadata(self) -> dict[str, tuple[TensorMetadata, ...]]:
        """Read actual OM names, dtypes and shapes in vendor execution order."""
        dtypes = {
            "float32": "float32",
            "float16": "float16",
            "double64": "float64",
            "float64": "float64",
            "int64": "int64",
            "int32": "int32",
            "uint8": "uint8",
            "bool": "bool",
        }
        metadata = {}
        for kind, nodes in (
            ("inputs", self.session.get_inputs()),
            ("outputs", self.session.get_outputs()),
        ):
            tensors = []
            for node in nodes:
                # ACLRuntime dtype is a pybind enum, not a NumPy dtype or ACL type ID.
                dtype = node.datatype.name
                if dtype not in dtypes:
                    raise ValueError(f"unsupported OM {kind} dtype: {node.name}: {dtype}")
                shape = tuple(node.shape)
                if not shape or any(not isinstance(d, int) or d <= 0 for d in shape):
                    raise ValueError(f"OM requires static positive {kind} dimensions: {node.name}")
                tensors.append(TensorMetadata(node.name, dtypes[dtype], shape))
            metadata[kind] = tuple(tensors)
        return metadata

    def _infer(self, tensors: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        inputs, outputs = self.config.inputs, self.config.outputs
        if inputs is None or outputs is None:
            raise ValueError("OM requires a resolved tensor contract")
        feeds = [np.ascontiguousarray(tensors[spec.name]) for spec in inputs]
        values = self.session.infer(feeds)
        if len(values) != len(outputs):
            raise ValueError("OM output count does not match configured outputs")
        result = {}
        for spec, value in zip(outputs, values):
            if not isinstance(value, np.ndarray):
                raise ValueError("OM outputs must be ndarrays")
            result[spec.name] = value
        return result
