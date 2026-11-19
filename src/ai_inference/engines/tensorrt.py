"""Native serialized TensorRT execution; CPU NumPy is the public tensor boundary."""

from collections.abc import Mapping
from math import prod
from threading import RLock

import numpy as np
from pydantic import Field

from ai_inference.config import BatchConfig, Config, Section
from ai_inference.contracts.execution import BatchCapabilities, ExecutionProfile, ShapeRange
from ai_inference.engines._cuda import CudaBuffer, CudaResources
from ai_inference.engines.base import BaseEngine, TensorMetadata
from ai_inference.engines.registry import EngineRegistry


class TensorRTOptions(Section):
    device_id: int = Field(default=0, ge=0, strict=True)


def _checked(success, operation):
    if not success:
        raise RuntimeError(f"TensorRT {operation} failed")


@EngineRegistry.register("tensorrt")
class TensorRTEngine(BaseEngine):
    def __init__(self, config: Config):
        super().__init__(config)
        self._lock = RLock()
        self._cuda = self._runtime = self._engine = self._context = None
        self._buffers = {}
        self._retired = []
        self._allocators = {}
        self._active_profile = None
        options = TensorRTOptions.model_validate(config.runtime.options)
        try:
            import tensorrt as trt
        except ImportError as exc:
            raise ImportError("Native TensorRT requires ai-inference[tensorrt-cu12]") from exc
        if trt.__version__.split(".")[0] != "10":
            raise RuntimeError("Native TensorRT backend currently requires TensorRT 10.x")
        self._trt = trt
        with self._initialization():
            self._cuda = CudaResources(options.device_id)
            with self._cuda.activate():
                self._logger = trt.Logger(trt.Logger.WARNING)
                _checked(trt.init_libnvinfer_plugins(self._logger, ""), "plugin initialization")
                self._runtime = trt.Runtime(self._logger)
                self._engine = self._runtime.deserialize_cuda_engine(config.model.path.read_bytes())
                if self._engine is None:
                    raise RuntimeError(
                        "TensorRT could not deserialize engine; check version/device compatibility"
                    )
                self._discover()
                self._resolve_batch_policy()
                self._load_tensor_contract()
                self._validate_policy()
                self._context = self._engine.create_execution_context()
                if self._context is None:
                    raise RuntimeError("TensorRT could not create execution context")
                self._create_allocators()

    def _discover(self):
        trt, engine = self._trt, self._engine
        metadata = {"inputs": [], "outputs": []}
        self._locations = {}
        count = max(1, engine.num_optimization_profiles)
        for index in range(engine.num_io_tensors):
            name = engine.get_tensor_name(index)
            if engine.is_shape_inference_io(name):
                raise ValueError(
                    f"TensorRT shape-inference I/O is outside the SDK batch contract: {name}"
                )
            for profile in range(count):
                if engine.get_tensor_format(name, profile) != trt.TensorFormat.LINEAR:
                    raise ValueError(f"TensorRT requires LINEAR I/O: {name}, profile={profile}")
            location = engine.get_tensor_location(name)
            if location not in (trt.TensorLocation.HOST, trt.TensorLocation.DEVICE):
                raise ValueError(f"unsupported TensorRT tensor location: {name}")
            self._locations[name] = location == trt.TensorLocation.DEVICE
            try:
                dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(name))).name
            except (TypeError, ValueError, KeyError) as exc:
                raise ValueError(f"unsupported TensorRT dtype: {name}") from exc
            if dtype not in {"float32", "float16", "float64", "int64", "int32", "uint8", "bool"}:
                raise ValueError(f"unsupported TensorRT dtype: {name}: {dtype}")
            dimensions = tuple(engine.get_tensor_shape(name))
            if any(dim != -1 and dim <= 0 for dim in dimensions):
                raise ValueError(
                    f"TensorRT requires positive or dynamic declared dimensions: {name}"
                )
            shape = tuple(int(dim) if dim > 0 else None for dim in dimensions)
            if not shape:
                raise ValueError(f"TensorRT tensors require a leading batch axis: {name}")
            kind = "inputs" if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT else "outputs"
            metadata[kind].append(TensorMetadata(name, dtype, shape))
        self._metadata = {kind: tuple(items) for kind, items in metadata.items()}
        profiles = []
        for index in range(count):
            inputs = []
            for tensor in self._metadata["inputs"]:
                if None in tensor.shape:
                    dimensions = engine.get_tensor_profile_shape(tensor.name, index)
                else:
                    dimensions = (tensor.shape,) * 3
                minimum, optimum, maximum = (tuple(map(int, dims)) for dims in dimensions)
                if any(dim <= 0 for dims in (minimum, optimum, maximum) for dim in dims):
                    raise ValueError(f"invalid TensorRT execution profile: {tensor.name}")
                inputs.append(ShapeRange(tensor.name, minimum, optimum, maximum))
            if not inputs:
                raise ValueError("TensorRT engine requires inputs")
            profiles.append(ExecutionProfile(tuple(inputs)))
        self.batch_capabilities = BatchCapabilities(tuple(profiles))

    def _resolve_batch_policy(self):
        if self.config.runtime.batch.mode != "auto":
            return
        inputs = self._metadata["inputs"]
        static = {tensor.shape[0] for tensor in inputs} - {None}
        padding = self.config.runtime.batch.padding
        if static:
            if len(static) != 1:
                raise ValueError("TensorRT inputs must share a leading batch dimension")
            policy = BatchConfig(mode="fixed", size=next(iter(static)), padding=padding)
        else:
            first = self.batch_capabilities.profiles[0]
            shapes = {item.name: item.optimum for item in first.inputs}
            bounds = first.batch_range(shapes)
            if bounds is None:
                raise ValueError("TensorRT profile has no common batch range")
            size = max(bounds[0], min(first.inputs[0].optimum[0], bounds[1]))
            policy = BatchConfig(mode="dynamic", size=size, padding=padding)
        self.config = self.config.model_copy(
            update={"runtime": self.config.runtime.model_copy(update={"batch": policy})}
        )

    def _validate_policy(self):
        policy = self.config.runtime.batch
        # Check explicit non-batch constraints without guessing unresolved dimensions.
        specs = {spec.name: spec for spec in self.config.inputs}
        for profile in self.batch_capabilities.profiles:
            shapes = {
                item.name: (
                    policy.size,
                    *(
                        dim if dim is not None else opt
                        for dim, opt in zip(specs[item.name].shape[1:], item.optimum[1:])
                    ),
                )
                for item in profile.inputs
            }
            bounds = profile.batch_range(shapes)
            if bounds and (
                bounds[0] <= policy.size <= bounds[1]
                if policy.mode == "fixed"
                else bounds[0] <= policy.size
            ):
                return
        raise ValueError("runtime.batch and input constraints match no TensorRT execution profile")

    def read_tensor_metadata(self):
        return self._metadata

    def create_warmup_inputs(self):
        tensors = super().create_warmup_inputs()
        if self.config.runtime.batch.mode == "dynamic":
            shapes = {name: value.shape for name, value in tensors.items()}
            ranges = [
                low
                for low, _ in self.batch_capabilities.ranges(shapes)
                if low <= self.config.runtime.batch.size
            ]
            if not ranges:
                raise ValueError("warmup input dimensions match no TensorRT profile")
            batch = min(ranges)
            tensors = {
                name: np.zeros((batch, *value.shape[1:]), value.dtype)
                for name, value in tensors.items()
            }
        return tensors

    def _buffer(self, name, size, alignment=256):
        old = self._buffers.get(name)
        if old is None or old.capacity < size or old.alignment < alignment:
            new = CudaBuffer(self._cuda, size, self._locations[name], alignment)
            self._buffers[name] = new
            if old is not None:
                self._retired.append(old)
        return self._buffers[name]

    def _create_allocators(self):
        owner = self

        class Allocator(self._trt.IOutputAllocator):
            def __init__(self):
                super().__init__()
                self.shape = self.error = None

            def reallocate_output(self, name, memory, size, alignment):
                try:
                    with owner._cuda.activate():
                        return owner._buffer(name, size, alignment).address
                except Exception as exc:
                    self.error = exc
                    return 0

            def reallocate_output_async(self, name, memory, size, alignment, stream):
                return self.reallocate_output(name, memory, size, alignment)

            def notify_shape(self, name, shape):
                self.shape = tuple(shape)

        for spec in self.config.outputs:
            allocator = Allocator()
            self._allocators[spec.name] = allocator
            _checked(
                self._context.set_output_allocator(spec.name, allocator), "set_output_allocator"
            )

    def _check_strides(self, name, shape):
        expected = 1
        strides = tuple(self._context.get_tensor_strides(name))
        if len(strides) != len(shape):
            raise ValueError(f"TensorRT tensor stride rank mismatch: {name}")
        for dim, stride in reversed(list(zip(shape, strides))):
            # Data-dependent extents may remain symbolic in the context. LINEAR
            # layout plus the allocator's notified shape defines those strides.
            if dim > 1 and stride not in (-1, expected):
                raise ValueError(f"TensorRT requires contiguous LINEAR tensor strides: {name}")
            expected *= dim

    def _infer(self, tensors: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        with self._lock:
            with self._cuda.activate():
                try:
                    return self._execute(tensors)
                finally:
                    # Keep buffers alive through failed enqueues/copies as well.
                    self._cuda.synchronize()
                    for buffer in self._retired:
                        buffer.close()
                    self._retired.clear()

    def _execute(self, tensors):
        context = self._context
        profile = self.batch_capabilities.select_profile(
            {name: value.shape for name, value in tensors.items()}
        )
        if profile != self._active_profile:
            _checked(
                context.set_optimization_profile_async(profile, int(self._cuda.stream)),
                "set_optimization_profile_async",
            )
            self._active_profile = profile
        for name, value in tensors.items():
            _checked(context.set_input_shape(name, value.shape), f"set_input_shape({name})")
            buffer = self._buffer(name, value.nbytes)
            buffer.upload(value)
            _checked(context.set_tensor_address(name, buffer.address), "set_tensor_address")
        missing = context.infer_shapes()
        if missing:
            raise ValueError(f"TensorRT could not resolve input shapes: {missing}")
        for name, value in tensors.items():
            self._check_strides(name, value.shape)
        for spec in self.config.outputs:
            allocator = self._allocators[spec.name]
            allocator.shape = allocator.error = None
            shape = tuple(context.get_tensor_shape(spec.name))
            address = 0
            if all(dim >= 0 for dim in shape):
                address = self._buffer(
                    spec.name, prod(shape) * np.dtype(spec.dtype).itemsize
                ).address
            _checked(context.set_tensor_address(spec.name, address), "set_tensor_address")
        success = context.execute_async_v3(int(self._cuda.stream))
        self._cuda.synchronize()
        for allocator in self._allocators.values():
            if allocator.error is not None:
                raise allocator.error
        _checked(success, "execute_async_v3")
        outputs = {}
        for spec in self.config.outputs:
            allocator = self._allocators[spec.name]
            shape = allocator.shape
            if shape is None:
                shape = tuple(context.get_tensor_shape(spec.name))
            if not shape or any(dim < 0 for dim in shape):
                raise RuntimeError(f"TensorRT output shape remains unresolved: {spec.name}")
            self._check_strides(spec.name, shape)
            size = prod(shape) * np.dtype(spec.dtype).itemsize
            buffer = self._buffers.get(spec.name)
            if buffer is None or size > buffer.capacity:
                raise RuntimeError(f"TensorRT did not allocate output: {spec.name}")
            buffer.download(size)
            outputs[spec.name] = (buffer, shape, spec.dtype)
        self._cuda.synchronize()
        return {
            name: buffer.array(shape, dtype).copy()
            for name, (buffer, shape, dtype) in outputs.items()
        }

    def _close(self):
        """Release owned CUDA/TRT resources after BaseEngine drains operations."""
        with self._lock:
            if self._cuda is None:
                return
            errors = []
            with self._cuda.activate():
                try:
                    self._cuda.synchronize()
                except Exception as exc:
                    errors.append(exc)
                self._context = None
                self._allocators.clear()
                for buffer in [*self._buffers.values(), *self._retired]:
                    try:
                        buffer.close()
                    except Exception as exc:
                        errors.append(exc)
                self._buffers.clear()
                self._retired.clear()
                self._engine = self._runtime = None
            try:
                self._cuda.close()
            except Exception as exc:
                errors.append(exc)
            if errors:
                raise errors[0]
