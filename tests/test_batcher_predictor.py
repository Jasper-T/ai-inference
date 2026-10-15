from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from ai_inference import Batcher, Config, DefaultPredictor, PreparedInput
from ai_inference.engines.base import BaseEngine


def contract(config, *, dynamic=False):
    data = config.model_dump()
    data["runtime"]["batch"]["mode"] = "dynamic" if dynamic else "fixed"
    batch = None if dynamic else 4
    data["inputs"] = [{"name": "x", "dtype": "float32", "shape": (batch, 2)}]
    data["outputs"] = [{"name": "y", "dtype": "float32", "shape": (batch, 2)}]
    return Config.model_validate(data)


class EchoEngine(BaseEngine):
    def _infer(self, tensors):
        return {"y": tensors["x"]}


def prepared(count, offset=0):
    x = np.arange(count * 2, dtype=np.float32).reshape(count, 2) + offset
    return PreparedInput({"x": x}, tuple(range(count)))


@pytest.mark.parametrize("dynamic,physical", [(False, [4, 4]), (True, [4, 1])])
def test_batcher_restores_samples_without_engine_dependency(config, dynamic, physical):
    batcher = Batcher(contract(config, dynamic=dynamic))
    request = prepared(5)
    chunks = list(batcher.split(request))
    assert [chunk.physical_count for chunk in chunks] == physical
    assert [chunk.start for chunk in chunks] == [0, 4]
    parts = []
    for chunk in chunks:
        buffer = chunk.tensors["x"].copy()
        parts.append(batcher.trim({"y": buffer}, chunk))
        buffer[:] = -1  # Backend buffer reuse cannot change collected output.
    raw = batcher.merge(parts, request)
    assert raw.prepared is request
    np.testing.assert_array_equal(raw.tensors["y"], request.tensors["x"])


@pytest.mark.parametrize("dynamic", [False, True])
def test_shared_batcher_predictor_keeps_concurrent_requests_independent(config, dynamic):
    config = contract(config, dynamic=dynamic)
    batcher = Batcher(config)
    predictor = DefaultPredictor(EchoEngine(config), config, batcher=batcher)
    requests = [prepared(count, index * 100) for index, count in enumerate((0, 1, 5, 9))]
    with ThreadPoolExecutor(max_workers=4) as pool:
        outputs = list(pool.map(predictor, requests))
    for raw, request in zip(outputs, requests):
        assert raw.prepared is request
        np.testing.assert_array_equal(raw.tensors["y"], request.tensors["x"])


def test_merge_rejects_missing_samples(config):
    batcher = Batcher(contract(config))
    request = prepared(5)
    first = next(batcher.split(request))
    with pytest.raises(ValueError, match="batch dimension"):
        batcher.merge([batcher.trim({"y": first.tensors["x"]}, first)], request)


def test_single_dynamic_sample_passes_without_padding(config):
    config = contract(config, dynamic=True)
    batcher = Batcher(config)
    request = prepared(1)
    [chunk] = batcher.split(request)
    assert chunk.valid_count == chunk.physical_count == 1
    assert chunk.padded_count == 0
    np.testing.assert_array_equal(
        DefaultPredictor(EchoEngine(config), config)(request).tensors["y"], request.tensors["x"]
    )
