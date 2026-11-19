"""Private CUDA Driver API ownership for the native TensorRT backend."""

import ctypes
from contextlib import contextmanager

import numpy as np


class CudaResources:
    def __init__(self, device_id: int):
        from cuda.bindings import driver

        self.driver = driver
        self.device = self.context = self.stream = None
        self.call("cuInit", 0)
        self.device = self.call("cuDeviceGet", device_id)
        self.context = self.call("cuDevicePrimaryCtxRetain", self.device)
        try:
            with self.activate():
                self.stream = self.call("cuStreamCreate", 1)  # CU_STREAM_NON_BLOCKING
        except Exception:
            self.close()
            raise

    def call(self, name, *args):
        status, *values = getattr(self.driver, name)(*args)
        if int(status) != 0:
            raise RuntimeError(f"CUDA {name} failed: {status}")
        return values[0] if len(values) == 1 else tuple(values)

    @contextmanager
    def activate(self):
        self.call("cuCtxPushCurrent", self.context)
        try:
            yield
        finally:
            self.call("cuCtxPopCurrent")

    def synchronize(self):
        self.call("cuStreamSynchronize", self.stream)

    def close(self):
        if self.context is None:
            return
        try:
            with self.activate():
                if self.stream is not None:
                    try:
                        self.synchronize()
                    finally:
                        self.call("cuStreamDestroy", self.stream)
                        self.stream = None
        finally:
            self.call("cuDevicePrimaryCtxRelease", self.device)
            self.context = None


class CudaBuffer:
    """Pinned host staging and optional device memory, with aligned tensor addresses."""

    def __init__(self, cuda: CudaResources, size: int, device: bool, alignment: int = 256):
        self.cuda = cuda
        self.capacity, self.alignment = max(size, 1), max(alignment, 256)
        self.host_raw = self.device_raw = None
        allocation = self.capacity + self.alignment - 1
        try:
            self.host_raw = cuda.call("cuMemHostAlloc", allocation, 0)
            self.host = self._align(int(self.host_raw))
            if device:
                self.device_raw = cuda.call("cuMemAlloc", allocation)
            self.address = self._align(int(self.device_raw)) if device else self.host
        except Exception:
            self.close()
            raise

    def _align(self, address):
        return (address + self.alignment - 1) // self.alignment * self.alignment

    def array(self, shape, dtype):
        count = int(np.prod(shape, dtype=object))
        size = count * np.dtype(dtype).itemsize
        if size > self.capacity:
            raise RuntimeError("TensorRT output exceeds allocated buffer")
        raw = (ctypes.c_ubyte * size).from_address(self.host)
        return np.frombuffer(raw, dtype=dtype, count=count).reshape(shape)

    def upload(self, value):
        np.copyto(self.array(value.shape, value.dtype), value)
        if self.device_raw is not None and value.nbytes:
            self.cuda.call(
                "cuMemcpyHtoDAsync", self.address, self.host, value.nbytes, self.cuda.stream
            )

    def download(self, size):
        if self.device_raw is not None and size:
            self.cuda.call("cuMemcpyDtoHAsync", self.host, self.address, size, self.cuda.stream)

    def close(self):
        try:
            if self.device_raw is not None:
                self.cuda.call("cuMemFree", self.device_raw)
                self.device_raw = None
        finally:
            if self.host_raw is not None:
                self.cuda.call("cuMemFreeHost", self.host_raw)
                self.host_raw = None
