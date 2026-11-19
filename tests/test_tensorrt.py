"""Exercise real backend ownership and batching against an in-memory CUDA/TRT double."""

import ctypes
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Event, get_ident
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from ai_inference import Config, DefaultPredictor, PreparedInput, TensorRTEngine
from ai_inference.engines._cuda import CudaBuffer, CudaResources


class Driver:
    def __init__(self):
        self.memory, self.stacks, self.queue = {}, {}, []
        self.retains = self.releases = self.streams = 0
        self.fail = None

    def cuInit(self, flags):
        return (0,)

    def cuDeviceGet(self, device):
        return (0, device)

    def cuDevicePrimaryCtxRetain(self, device):
        self.retains += 1
        return (0, 42 + device)

    def cuDevicePrimaryCtxRelease(self, device):
        self.releases += 1
        return (0,)

    def cuCtxPushCurrent(self, context):
        self.stacks.setdefault(get_ident(), []).append(context)
        return (0,)

    def cuCtxPopCurrent(self):
        return (0, self.stacks[get_ident()].pop())

    def cuStreamCreate(self, flags):
        assert self.stacks[get_ident()]
        if self.fail == "stream":
            return (1, None)
        self.streams += 1
        return (0, 100)

    def cuStreamDestroy(self, stream):
        self.streams -= 1
        return (0,)

    def cuStreamSynchronize(self, stream):
        assert self.stacks[get_ident()]
        while self.queue:
            self.queue.pop(0)()
        return (0,)

    def cuMemHostAlloc(self, size, flags):
        assert self.stacks[get_ident()]
        data = ctypes.create_string_buffer(size)
        pointer = ctypes.addressof(data)
        self.memory[pointer] = data
        return (0, pointer)

    def cuMemAlloc(self, size):
        if self.fail == "allocation":
            return (1, None)
        return self.cuMemHostAlloc(size, 0)

    def cuMemFree(self, pointer):
        del self.memory[pointer]
        return (0,)

    cuMemFreeHost = cuMemFree

    def cuMemcpyHtoDAsync(self, dst, src, size, stream):
        self.queue.append(lambda: ctypes.memmove(dst, src, size))
        return (0,)

    cuMemcpyDtoHAsync = cuMemcpyHtoDAsync


@pytest.fixture
def backend(monkeypatch, tmp_path):
    driver = Driver()
    cuda, bindings = ModuleType("cuda"), ModuleType("cuda.bindings")
    bindings.driver = driver
    cuda.bindings = bindings
    monkeypatch.setitem(sys.modules, "cuda", cuda)
    monkeypatch.setitem(sys.modules, "cuda.bindings", bindings)
    trt = ModuleType("tensorrt")
    trt.__version__ = "10.16.0.72"
    trt.Logger = type("Logger", (), {"WARNING": 1, "__init__": lambda self, level: None})
    trt.TensorFormat = SimpleNamespace(LINEAR=0)
    trt.TensorLocation = SimpleNamespace(HOST=0, DEVICE=1)
    trt.TensorIOMode = SimpleNamespace(INPUT=0, OUTPUT=1)
    trt.nptype = lambda dtype: dtype
    trt.IOutputAllocator = type("IOutputAllocator", (), {})
    trt.init_libnvinfer_plugins = lambda *args: True

    class Context:
        def __init__(self):
            self.shapes, self.addresses, self.allocators = {}, {}, {}
            self.batches, self.profiles = [], []
            self.output_shape = None

        def set_optimization_profile_async(self, profile, stream):
            self.profiles.append(profile)
            return True

        def set_input_shape(self, name, shape):
            self.shapes[name] = shape
            return True

        def set_tensor_address(self, name, address):
            self.addresses[name] = address
            return True

        def infer_shapes(self):
            return []

        def get_tensor_shape(self, name):
            if engine.dependent:
                return (-1, -1)
            return self.shapes["x"]

        def get_tensor_strides(self, name):
            return (2, 1)

        def set_output_allocator(self, name, allocator):
            self.allocators[name] = allocator
            return True

        def execute_async_v3(self, stream):
            shape = self.shapes["x"]
            self.batches.append(shape[0])
            allocator = self.allocators["y"]
            if engine.dependent:
                address = allocator.reallocate_output_async(
                    "y", 0, int(np.prod(shape)) * 4, 512, stream
                )
                if not address:
                    return False
                self.addresses["y"] = address
                allocator.notify_shape("y", shape)

            def execute():
                src = (ctypes.c_float * np.prod(shape)).from_address(self.addresses["x"])
                dst = (ctypes.c_float * np.prod(shape)).from_address(self.addresses["y"])
                np.copyto(np.ctypeslib.as_array(dst), np.ctypeslib.as_array(src) * 2)

            driver.queue.append(execute)
            return not engine.fail_execute

    class Engine:
        num_optimization_profiles = 1
        num_io_tensors = 2
        dependent, fail_execute = False, False
        static = None
        format = 0
        location = 1
        dtype = np.float32
        shape_io = False

        def __init__(self):
            self.context = Context()
            self.profiles = [((2, 2), (4, 2), (8, 2))]

        def get_tensor_name(self, index):
            return ("x", "y")[index]

        def is_shape_inference_io(self, name):
            return self.shape_io

        def get_tensor_format(self, name, profile):
            return self.format

        def get_tensor_location(self, name):
            return self.location

        def get_tensor_dtype(self, name):
            return self.dtype

        def get_tensor_shape(self, name):
            return (self.static or -1, -1 if name == "y" and self.dependent else 2)

        def get_tensor_mode(self, name):
            return 0 if name == "x" else 1

        def get_tensor_profile_shape(self, name, profile):
            return self.profiles[profile]

        def create_execution_context(self):
            return self.context

    engine = Engine()
    trt.Runtime = lambda logger: SimpleNamespace(deserialize_cuda_engine=lambda data: engine)
    monkeypatch.setitem(sys.modules, "tensorrt", trt)
    path = tmp_path / "test.engine"
    path.write_bytes(b"serialized")

    def create(**runtime):
        return TensorRTEngine(
            Config(
                model={"name": "test", "path": path},
                runtime={"backend": "tensorrt", **runtime},
                preprocess={"steps": []},
                postprocess={"steps": []},
            )
        )

    return SimpleNamespace(create=create, driver=driver, engine=engine, trt=trt)


def request(count):
    values = np.arange(count * 2, dtype=np.float32).reshape(count, 2)
    return PreparedInput({"x": values}, tuple(range(count)))


def test_native_dynamic_minimum_batch_warmup_and_tail(backend):
    engine = backend.create()
    assert engine.config.runtime.batch.mode == "dynamic"
    assert engine.config.runtime.batch.size == 4
    assert engine.read_tensor_metadata()["inputs"][0].shape == (None, 2)
    engine.warmup()
    assert backend.engine.context.batches == [2] * 3
    prepared = request(5)
    raw = DefaultPredictor(engine, engine.config)(prepared)
    np.testing.assert_array_equal(raw.tensors["y"], prepared.tensors["x"] * 2)
    assert backend.engine.context.batches[-2:] == [4, 2]
    with pytest.raises(ValueError, match="jointly match"):
        engine.infer(request(1).tensors)
    engine.close()
    engine.close()
    assert not backend.driver.memory and backend.driver.streams == 0
    assert backend.driver.retains == backend.driver.releases == 1
    with pytest.raises(RuntimeError, match="closed"):
        engine.infer(request(2).tensors)


@pytest.mark.parametrize("location", [0, 1])
def test_data_dependent_output_host_and_device_reuse_and_growth(backend, location):
    backend.engine.dependent = True
    backend.engine.location = location
    engine = backend.create(batch={"mode": "dynamic", "size": 8})
    first = engine.infer(request(2).tensors)["y"]
    second = engine.infer(request(8).tensors)["y"]
    np.testing.assert_array_equal(first, request(2).tensors["x"] * 2)
    np.testing.assert_array_equal(second, request(8).tensors["x"] * 2)
    assert not engine._retired
    assert engine._buffers["y"].address % 512 == 0
    engine.close()
    assert not backend.driver.memory


def test_static_default_does_not_inherit_sdk_batch_four(backend):
    backend.engine.static = 2
    engine = backend.create()
    assert engine.config.runtime.batch.mode == "fixed"
    assert engine.config.runtime.batch.size == 2
    prepared = request(3)
    raw = DefaultPredictor(engine, engine.config)(prepared)
    np.testing.assert_array_equal(raw.tensors["y"], prepared.tensors["x"] * 2)
    assert backend.engine.context.batches == [2, 2]
    engine.close()


@pytest.mark.parametrize("batch", [{"mode": "fixed", "size": 1}, {"mode": "dynamic", "size": 1}])
def test_incompatible_explicit_policy_cleans_constructor(backend, batch):
    with pytest.raises(ValueError, match="profile"):
        backend.create(batch=batch)
    assert backend.driver.releases == 1 and backend.driver.streams == 0


@pytest.mark.parametrize("attribute,value", [("format", 2), ("dtype", np.int8), ("shape_io", True)])
def test_unsupported_artifact_fails_before_context_creation(backend, attribute, value):
    setattr(backend.engine, attribute, value)
    with pytest.raises(ValueError):
        backend.create()
    assert backend.driver.releases == 1


def test_execute_failure_synchronizes_and_can_close(backend):
    engine = backend.create()
    backend.engine.fail_execute = True
    with pytest.raises(RuntimeError, match="execute_async_v3"):
        engine.infer(request(2).tensors)
    assert not backend.driver.queue
    engine.close()
    assert not backend.driver.memory


def test_profile_switch_and_allocator_failure(backend, monkeypatch):
    backend.engine.num_optimization_profiles = 2
    backend.engine.profiles = [((2, 2), (2, 2), (3, 2)), ((6, 2), (6, 2), (8, 2))]
    backend.engine.dependent = True
    engine = backend.create(batch={"mode": "dynamic", "size": 8})
    engine.infer(request(2).tensors)
    engine.infer(request(6).tensors)
    assert backend.engine.context.profiles == [0, 1]
    original = engine._buffer

    def fail_output(name, *args):
        if name == "y":
            raise MemoryError("output allocation failed")
        return original(name, *args)

    monkeypatch.setattr(engine, "_buffer", fail_output)
    with pytest.raises(MemoryError, match="output allocation failed"):
        engine.infer(request(2).tensors)
    engine.close()
    assert not backend.driver.memory


def test_auto_policy_preserves_explicit_padding(backend):
    engine = backend.create(batch={"mode": "auto", "padding": "zeros"})
    assert engine.config.runtime.batch.padding == "zeros"
    engine.close()


def test_buffer_and_stream_partial_allocation_cleanup(backend):
    backend.driver.fail = "stream"
    with pytest.raises(RuntimeError, match="cuStreamCreate"):
        CudaResources(0)
    assert backend.driver.releases == 1
    backend.driver.fail = None
    resources = CudaResources(0)
    with resources.activate():
        backend.driver.fail = "allocation"
        with pytest.raises(RuntimeError, match="cuMemAlloc"):
            CudaBuffer(resources, 20, True)
        assert not backend.driver.memory
    resources.close()


def test_context_thread_switch_and_close_waits_for_execute(backend, monkeypatch):
    engine = backend.create()
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(engine.infer, request(2).tensors).result(timeout=5)
    np.testing.assert_array_equal(result["y"], request(2).tensors["x"] * 2)
    assert all(not stack for stack in backend.driver.stacks.values())
    entered, release, close_started = Event(), Event(), Event()
    execute = engine._execute

    def blocked(tensors):
        entered.set()
        assert release.wait(5)
        return execute(tensors)

    monkeypatch.setattr(engine, "_execute", blocked)

    def close():
        close_started.set()
        engine.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        run = pool.submit(engine.infer, request(2).tensors)
        assert entered.wait(5)
        closing = pool.submit(close)
        assert close_started.wait(5)
        assert not closing.done()
        release.set()
        run.result(timeout=5)
        closing.result(timeout=5)
    assert not backend.driver.memory and backend.driver.releases == 1
