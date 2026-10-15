"""Synthetic contracts: no pretrained weights or GPU claims."""

from pathlib import Path

import cv2
import numpy as np
import pytest
from model_contracts import load_synthetic_config
from pipeline_helpers import options, pipelines

from ai_inference import (
    Config,
    EngineRegistry,
    FaceInput,
    FeaturePrediction,
    Model,
    Prediction,
    RawOutput,
    Result,
)
from ai_inference.engines.base import BaseEngine
from ai_inference.processing.decoders.sface import TARGET_POINTS, align_face

ROOT = Path(__file__).resolve().parents[1]


def cfg(name):
    family = {"yunet_2023": "yunet", "sface_2021": "sface"}.get(name, name)
    path = (
        ROOT / "tests/fixtures/rfdetr_synthetic.yaml"
        if family == "rfdetr"
        else ROOT / "configs" / f"{family}.yaml"
    )
    data = load_synthetic_config(path).model_dump()
    data["runtime"]["warmup"] = True  # Exercise warmup independently of demo defaults.
    data["preprocess"]["source_color"] = "RGB"
    options(data, "color")["source"] = "RGB"  # color must agree with the source contract
    return Config.model_validate(data)


def pair(name):
    c = cfg(name)
    return c, pipelines(c)


def raw(c, prepared):
    return {s.name: np.zeros((prepared.count, *s.shape[1:]), dtype=s.dtype) for s in c.outputs}


def test_prediction_and_mask_extension():
    old = Prediction(np.empty((0, 4)), np.empty(0), np.empty(0), (10, 20))
    assert old.keypoints is None and old.masks is None
    extended = Prediction(
        old.boxes,
        old.scores,
        old.class_ids,
        old.original_shape,
        masks=np.empty((0, 10, 20), np.uint8),
    )
    assert extended.masks is not None
    assert extended.masks.shape == (0, 10, 20)


def test_rtdetr_pixels_sizes_and_original_boxes():
    c, a = pair("rtdetr")
    image = np.full((100, 200, 3), [255, 100, 0], np.uint8)
    p = a.preprocess(image)
    np.testing.assert_array_equal(p.tensors["orig_target_sizes"], [[200, 100]])
    np.testing.assert_allclose(p.tensors["images"][0, :, 0, 0], [1, 100 / 255, 0])
    t = raw(c, p)
    t["scores"][0, :3] = [0.8, 0.9, 0.2]
    t["labels"][0, 1] = 1
    t["boxes"][0, 0] = [10, 20, 60, 80]
    result = a.postprocess(RawOutput(t, p))[0]
    np.testing.assert_array_equal(result.boxes, [[10, 20, 60, 80], [0, 0, 0, 0]])
    assert result.class_ids.tolist() == [0, 1]


def test_rtdetr_padding_rejected_before_engine(monkeypatch):
    data = cfg("rtdetr").model_dump()
    options(data, "resize").update(width=320, height=320)
    data["preprocess"]["steps"].insert(2, {"padding": {"width": 640, "height": 640}})

    def unexpected(*args):
        pytest.fail("invalid geometry must not allocate an engine")

    monkeypatch.setattr(EngineRegistry, "create", unexpected)
    with pytest.raises(ValueError, match="direct resize without padding"):
        Model(Config.model_validate(data))


def test_rtdetr_rejects_injected_padded_metadata():
    from dataclasses import replace

    from ai_inference import PreparedInput

    c, a = pair("rtdetr")
    p = a.preprocess(np.zeros((80, 100, 3), np.uint8))
    padded = PreparedInput(p.tensors, (replace(p.metadata[0], padding=(10, 20)),))
    with pytest.raises(ValueError, match="direct resize without padding"):
        a.postprocess(RawOutput(raw(c, p), padded))


@pytest.mark.parametrize("mutation", ["nan", "label", "reversed", "score"])
def test_rtdetr_invalid_outputs(mutation):
    c, a = pair("rtdetr")
    p = a.preprocess(np.zeros((80, 100, 3), np.uint8))
    t = raw(c, p)
    if mutation == "nan":
        t["boxes"][0, 0, 0] = np.nan
    if mutation == "label":
        t["labels"][0, 0] = 80
    if mutation == "reversed":
        t["boxes"][0, 0] = [5, 5, 1, 1]
    if mutation == "score":
        t["scores"][0, 0] = 1.1
    with pytest.raises(ValueError):
        a.postprocess(RawOutput(t, p))


def test_yunet_bgr_raw_pixels_decode_and_nms():
    c, a = pair("yunet_2023")
    p = a.preprocess(np.full((640, 160, 3), [1, 2, 3], np.uint8))
    np.testing.assert_array_equal(p.tensors["input"][0, :, 0, 0], [3, 2, 1])
    t = raw(c, p)
    for stride in (8, 16, 32):
        index = 2 * (640 // stride) + 3
        t[f"cls_{stride}"][0, index, 0] = 1
        t[f"obj_{stride}"][0, index, 0] = 0.81
        t[f"bbox_{stride}"][0, index] = [0, 0, 0, 0]
    result = a.postprocess(RawOutput(t, p))[0]
    assert result.keypoints.shape == (3, 5, 2)
    np.testing.assert_allclose(result.boxes[0], [5, 12, 7, 20])
    np.testing.assert_allclose(result.keypoints[0], [[6, 16]] * 5)
    # Two stride heads describe the same box; suppress lower score.
    t = raw(c, p)
    for stride, score in ((8, 0.9), (16, 0.85)):
        t[f"cls_{stride}"][0, 0, 0] = score**2
        t[f"obj_{stride}"][0, 0, 0] = 1
        t[f"bbox_{stride}"][0, 0] = [
            32 / stride,
            32 / stride,
            np.log(16 / stride),
            np.log(16 / stride),
        ]
    assert len(a.postprocess(RawOutput(t, p))[0].boxes) == 1


def test_yunet_empty_and_nonfinite():
    c, a = pair("yunet_2023")
    p = a.preprocess(np.zeros((320, 320, 3), np.uint8))
    t = raw(c, p)
    result = a.postprocess(RawOutput(t, p))[0]
    assert result.boxes.shape == (0, 4) and result.keypoints.shape == (0, 5, 2)
    t["kps_8"][0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        a.postprocess(RawOutput(t, p))


def test_sface_alignment_raw_pixels_and_features():
    c, a = pair("sface_2021")
    image = np.full((112, 112, 3), [250, 120, 10], np.uint8)
    p = a.preprocess(FaceInput(image, TARGET_POINTS))
    np.testing.assert_array_equal(p.tensors["data"][0, :, 50, 50], [250, 120, 10])
    t = raw(c, p)
    t["fc1"][0, :2] = [3, 4]
    result = a.postprocess(RawOutput(t, p))[0]
    assert isinstance(result, FeaturePrediction)
    np.testing.assert_allclose(result.embedding[:2], [0.6, 0.8])
    assert result.embedding.dtype == np.float32
    with pytest.raises(TypeError):
        a.preprocess(image)
    with pytest.raises(ValueError):
        a.preprocess(FaceInput(image, np.zeros((5, 2))))
    t["fc1"][:] = 0
    with pytest.raises(ValueError):
        a.postprocess(RawOutput(t, p))
    t["fc1"][0, 0] = np.inf
    with pytest.raises(ValueError):
        a.postprocess(RawOutput(t, p))


def test_similarity_alignment_transformed_landmarks():
    rng = np.random.default_rng(9)
    image = rng.integers(0, 256, (112, 112, 3), np.uint8)
    np.testing.assert_array_equal(align_face(image, TARGET_POINTS), image)
    # Enlarge and offset the original image and landmarks; aligned center agrees.
    large = np.zeros((244, 254, 3), np.uint8)
    large[20:244, 30:254] = cv2.resize(image, (224, 224))
    aligned = align_face(large, TARGET_POINTS * 2 + [30, 20])
    assert aligned.shape == image.shape
    assert np.abs(aligned.astype(float)[10:100] - image.astype(float)[10:100]).mean() < 55


@pytest.mark.parametrize("name", ["rtdetr", "yunet_2023", "sface_2021"])
def test_contract_rejects_wrong_output(name):
    c = cfg(name)
    data = c.model_dump()
    data["outputs"][0]["name"] = "wrong"
    with pytest.raises(ValueError):
        pipelines(Config.model_validate(data))


@pytest.mark.parametrize("name", ["rtdetr", "yunet_2023", "sface_2021"])
@pytest.mark.parametrize("mode", ["fixed", "dynamic"])
def test_model_batch_warmup_and_empty(name, mode):
    c = cfg(name)
    data = c.model_dump()
    data["runtime"]["batch"].update(mode=mode, size=2)
    for spec in data["inputs"] + data["outputs"]:
        spec["shape"] = (2 if mode == "fixed" else None, *spec["shape"][1:])
    calls = []

    class SyntheticEngine(BaseEngine):
        def _infer(self, tensors):
            n = next(iter(tensors.values())).shape[0]
            calls.append(n)
            specs = self.config.outputs
            assert specs is not None
            t = {}
            for spec in specs:
                assert spec.shape is not None
                t[spec.name] = np.zeros((n, *(d or 1 for d in spec.shape[1:])), spec.dtype)
            if name == "sface_2021":
                t["fc1"][:, 0] = 1
            return t

    key = f"extended-test-{name}-{mode}"
    EngineRegistry.register(key, SyntheticEngine)
    data["runtime"]["backend"] = key
    model = Model(Config.model_validate(data))
    assert len(calls) == 3
    image = np.zeros((112, 112, 3), np.uint8)
    results = (
        model.predict([FaceInput(image, TARGET_POINTS)] * 3)
        if name == "sface_2021"
        else model.predict([image] * 3)
    )
    assert len(results) == 3
    assert calls[-2:] == ([2, 2] if mode == "fixed" else [2, 1])
    before = len(calls)
    assert model.predict([]) == []
    assert len(calls) == before


def test_yaml_disallows_script_settings():
    c = cfg("sface_2021")
    d = c.model_dump()
    d["preprocess"]["script"] = "anything.py"
    with pytest.raises(ValueError):
        pipelines(Config.model_validate(d))


@pytest.mark.parametrize("name", ["rtdetr", "yunet_2023", "sface_2021"])
def test_real_ort_synthetic_contract(tmp_path, name):
    """Actual CPU ORT executes constant-output ONNX; no trained model evidence."""
    import onnx
    from onnx import helper, numpy_helper

    c = cfg(name)
    d = c.model_dump()
    graph_inputs = []
    graph_outputs = []
    nodes = []
    assert c.inputs is not None and c.outputs is not None
    for spec in c.inputs:
        assert spec.shape is not None
        dtype = helper.np_dtype_to_tensor_dtype(np.dtype(spec.dtype))
        graph_inputs.append(helper.make_tensor_value_info(spec.name, dtype, list(spec.shape)))
    for spec in c.outputs:
        assert spec.shape is not None
        values = np.zeros(tuple(d or 1 for d in spec.shape), spec.dtype)
        if name == "sface_2021":
            values[:, 0] = 1
        if name == "rtdetr" and spec.name == "scores":
            values[:, 0] = 0.9
        if name == "rtdetr" and spec.name == "boxes":
            values[:, 0] = [1, 2, 30, 50]
        nodes.append(
            helper.make_node("Constant", [], [spec.name], value=numpy_helper.from_array(values))
        )
        graph_outputs.append(
            helper.make_tensor_value_info(
                spec.name, helper.np_dtype_to_tensor_dtype(values.dtype), list(spec.shape)
            )
        )
    graph = helper.make_graph(nodes, "synthetic-contract", graph_inputs, graph_outputs)
    onnx_model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx_model.ir_version = 8
    path = tmp_path / "synthetic.onnx"
    onnx.save(onnx_model, path)
    d["model"].update(path=path)
    d["runtime"]["options"]["providers"] = "cpu"
    model = Model(Config.model_validate(d))
    image = np.zeros((112, 112, 3), np.uint8)
    results = (
        model.predict([FaceInput(image, TARGET_POINTS)] * 3)
        if name == "sface_2021"
        else model.predict([image] * 3)
    )
    assert len(results) == 3
    assert model.engine.providers == ("CPUExecutionProvider",)
    if name == "rtdetr":
        assert isinstance(results[0], Prediction)
        np.testing.assert_array_equal(results[0].boxes, [[1, 2, 30, 50]])
    if name == "sface_2021":
        assert isinstance(results[0], FeaturePrediction)
        assert results[0].embedding[0] == 1
    if name == "yunet_2023":
        assert isinstance(results[0], Prediction) and results[0].keypoints is not None
        assert results[0].keypoints.shape == (0, 5, 2)


def test_injected_mask_postprocessor_uses_shared_pipeline():
    c = cfg("yunet_2023")
    d = c.model_dump()
    d["runtime"].update(backend="mask-test-engine", warmup=False)

    class E(BaseEngine):
        def _infer(self, tensors):
            n = next(iter(tensors.values())).shape[0]
            specs = self.config.outputs
            assert specs is not None
            result = {}
            for spec in specs:
                assert spec.shape is not None
                result[spec.name] = np.zeros((n, *(d or 1 for d in spec.shape[1:])), spec.dtype)
            return result

    EngineRegistry.register("mask-test-engine", E)

    def post(raw) -> list[Result]:
        return [
            Prediction(
                np.array([[0, 0, 10, 10]], np.float32),
                np.array([1], np.float32),
                np.array([0]),
                m.original_shape,
                masks=np.ones((1, *m.original_shape), np.uint8),
            )
            for m in raw.prepared.metadata
        ]

    model = Model(Config.model_validate(d), postprocessor=post)
    prediction = model.predict(np.zeros((20, 30, 3), np.uint8))[0]
    assert isinstance(prediction, Prediction) and prediction.masks is not None
    assert prediction.masks.shape == (1, 20, 30)


def test_yunet_coordinate_overflow_rejected():
    c, a = pair("yunet_2023")
    p = a.preprocess(np.zeros((320, 320, 3), np.uint8))
    t = raw(c, p)
    t["cls_8"][0, 0, 0] = 1
    t["obj_8"][0, 0, 0] = 1
    t["kps_8"][0, 0, :] = np.finfo(np.float32).max
    with pytest.raises(ValueError, match="float32 range"):
        a.postprocess(RawOutput(t, p))


def test_sface_aligned_crop_opt_in():
    c = cfg("sface_2021")
    d = c.model_dump()
    options(d, "face_align").update(aligned_input=True)
    d["preprocess"]["source_color"] = "BGR"
    options(d, "color")["source"] = "BGR"
    a = pipelines(Config.model_validate(d))
    crop = np.full((112, 112, 3), [10, 20, 30], np.uint8)
    np.testing.assert_array_equal(a.preprocess(crop).tensors["data"][0, :, 0, 0], [30, 20, 10])
    with pytest.raises(ValueError):
        a.preprocess(np.zeros((100, 100, 3), np.uint8))


def test_yunet_matches_official_2023mar_static_contract():
    c = cfg("yunet_2023")
    assert c.inputs is not None and c.outputs is not None
    assert c.inputs[0].shape == (1, 3, 640, 640)
    assert options(c, "resize")["height"] == options(c, "resize")["width"] == 640
    for spec in c.outputs:
        assert spec.shape is not None
        stride = int(spec.name.rsplit("_", 1)[1])
        assert spec.shape[1] == (640 // stride) ** 2


@pytest.mark.parametrize("backend,pixel", [("pillow", 126), ("opencv", 96), (None, 126)])
def test_rtdetr_antialias_downsample_contract(backend, pixel):
    # Fixed checkerboard golden: official PIL bilinear antialias vs OpenCV.
    c = cfg("rtdetr")
    d = c.model_dump()
    options(d, "resize").update(height=2, width=2)
    if backend is None:
        options(d, "resize").update(backend="pillow")
    else:
        options(d, "resize")["backend"] = backend
    d["inputs"][0]["shape"] = (1, 3, 2, 2)
    a = pipelines(Config.model_validate(d))
    checker = (np.indices((5, 5)).sum(0) % 2 * 255).astype(np.uint8)
    image = np.repeat(checker[:, :, None], 3, axis=2)
    p = a.preprocess(image)
    np.testing.assert_allclose(p.tensors["images"], pixel / 255, atol=1e-7)
    np.testing.assert_array_equal(p.tensors["orig_target_sizes"], [[5, 5]])
    assert p.metadata[0].original_shape == (5, 5)


@pytest.mark.parametrize("family", ["rtdetr", "rfdetr", "yolo11"])
@pytest.mark.parametrize("count", [0, True])
def test_num_classes_requires_positive_integer(family, count):
    c = cfg(family)
    d = c.model_dump()
    next(iter(d["postprocess"]["steps"][0].values()))["num_classes"] = count
    with pytest.raises(ValueError):
        pipelines(Config.model_validate(d))
