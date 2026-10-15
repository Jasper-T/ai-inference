import numpy as np
import pytest
from pipeline_helpers import pipelines

from ai_inference import BaseEngine, Config, EngineRegistry, Model, Prediction, PreparedInput
from ai_inference.contracts.registry import Registry
from ai_inference.inference.predictor import DefaultPredictor


class Recorder(BaseEngine):
    def __init__(self, config):
        self.config = config
        self.calls = []

    def _infer(self, tensors):
        self.calls.append({k: v.copy() for k, v in tensors.items()})
        n = len(next(iter(tensors.values())))
        boxes = np.tile(np.array([0.5, 0.5, 0.5, 0.5], np.float32), (n, 300, 1))
        logits = np.full((n, 300, 2), -100, np.float32)
        logits[:, 0, 0] = 100
        return {"dets": boxes, "labels": logits}


EngineRegistry.register("test_recorder", Recorder)


def changed(config, **runtime):
    data = config.model_dump()
    data["runtime"].update(runtime)
    return Config.model_validate(data)


@pytest.mark.parametrize(
    "n,counts", [(0, []), (1, [4]), (3, [4]), (4, [4]), (5, [4, 4]), (9, [4, 4, 4])]
)
def test_fixed_batch_and_order(config, n, counts):
    model = Model(changed(config, backend="test_recorder"))
    images = [np.full((10 + i, 20 + i, 3), i, np.uint8) for i in range(n)]
    prepared = model.preprocess(images)
    raw = model.infer(prepared)
    results = model.postprocess(raw)
    assert [len(c["input"]) for c in model.engine.calls] == counts
    assert len(results) == n
    assert [p.original_shape for p in results] == [(10 + i, 20 + i) for i in range(n)]
    if n:
        np.testing.assert_array_equal(raw.tensors["labels"].shape, (n, 300, 2))
        for prediction in results:
            assert isinstance(prediction, Prediction)
            assert prediction.class_ids.tolist() == [0]
        if n % 4:
            last = model.engine.calls[-1]["input"]
            for padded in last[n % 4 :]:
                np.testing.assert_array_equal(padded, last[n % 4 - 1])


def test_staged_equals_predict_and_metadata_isolation(config):
    model = Model(changed(config, backend="test_recorder"))
    image = np.zeros((100, 200, 3), np.uint8)
    a = model.preprocess(image)
    b = model.preprocess(np.zeros((20, 30, 3), np.uint8))
    raw = model.infer(a)
    model.infer(b)
    result = model.postprocess(raw)[0]
    predicted = model.predict(image)[0]
    assert isinstance(result, Prediction) and isinstance(predicted, Prediction)
    np.testing.assert_array_equal(result.boxes, predicted.boxes)
    np.testing.assert_array_equal(result.boxes, [[50, 25, 150, 75]])
    with pytest.raises(TypeError):
        model.postprocess(raw, b)


def test_dynamic_and_zero_padding(config):
    for mode, padding, expected in [("dynamic", "repeat_last", [4, 1]), ("fixed", "zeros", [4, 4])]:
        data = config.model_dump()
        data["runtime"]["backend"] = "test_recorder"
        data["runtime"]["batch"].update(mode=mode, padding=padding)
        if mode == "dynamic":
            for key in ("inputs", "outputs"):
                data[key] = [dict(s, shape=(None, *s["shape"][1:])) for s in data[key]]
        model = Model(Config.model_validate(data))
        assert len(model.predict([np.ones((3, 3, 3), np.uint8)] * 5)) == 5
        assert [len(c["input"]) for c in model.engine.calls] == expected
        if padding == "zeros":
            assert not model.engine.calls[-1]["input"][1:].any()


def test_multiple_inputs_and_output_buffer_reuse(config):
    data = config.model_dump()
    data["inputs"] = [
        dict(name="a", dtype="float32", shape=(4, 1)),
        dict(name="b", dtype="int64", shape=(4, 1)),
    ]
    data["outputs"] = [dict(name="sum", dtype="float32", shape=(4, 1))]
    config = Config.model_validate(data)

    class MultiEngine(BaseEngine):
        buffer = np.empty((4, 1), np.float32)

        def infer(self, tensors):
            self.buffer[:] = tensors["a"] + tensors["b"]
            return {"sum": self.buffer}

    x = np.arange(7).reshape(7, 1)
    prepared = PreparedInput({"a": x.astype(np.float32), "b": x.astype(np.int64)}, tuple(range(7)))
    raw = DefaultPredictor(MultiEngine(config), config)(prepared)
    np.testing.assert_array_equal(raw.tensors["sum"], x * 2)


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((4, 4), np.uint8),
        np.zeros((2, 2, 3), np.float32),
        np.zeros((0, 2, 3), np.uint8),
        np.zeros((1, 2, 2, 3), np.uint8),
        "base64",
    ],
)
def test_bad_images(config, bad):
    with pytest.raises((ValueError, TypeError)):
        pipelines(config).preprocess(bad)


def test_contract_validation(config):
    prep = pipelines(config).preprocess(np.zeros((3, 3, 3), np.uint8))
    with pytest.raises(ValueError):
        DefaultPredictor(Recorder(config), config)(
            PreparedInput({"wrong": prep.tensors["input"]}, prep.metadata)
        )

    class BadEngine(BaseEngine):
        def infer(self, tensors):
            return {
                "dets": np.zeros((1, 300, 4), np.float32),
                "labels": np.zeros((1, 300, 2), np.float32),
            }

    with pytest.raises(ValueError):
        DefaultPredictor(BadEngine(config), config)(prep)
    data = config.model_dump()
    data["runtime"]["batch"]["size"] = 2
    with pytest.raises(ValueError):
        Config.model_validate(data)
    data = config.model_dump()
    data["runtime"]["typo"] = 1
    with pytest.raises(ValueError):
        Config.model_validate(data)


def test_registry():
    registry = Registry("test")
    registry.register("a", lambda config: config)
    assert registry.create("a", 3) == 3
    with pytest.raises(ValueError):
        registry.register("a", lambda c: c)
    with pytest.raises(ValueError):
        registry.create("missing", None)


def test_yaml_relative_path(config, tmp_path):
    import yaml

    data = config.model_dump(mode="json")
    data["model"]["path"] = "../models/model.onnx"
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "model.yaml"
    config_path.write_text(yaml.safe_dump(data))
    loaded = Config.from_yaml(config_path)
    assert loaded.model.path == (tmp_path / "models" / "model.onnx").resolve()
