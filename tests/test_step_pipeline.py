"""Default pipelines, model contracts and request isolation."""

import logging
from pathlib import Path

import numpy as np
import pytest
from model_contracts import load_synthetic_config

from ai_inference import (
    BaseEngine,
    Config,
    Context,
    EngineRegistry,
    FaceInput,
    FeaturePrediction,
    Model,
    Pipeline,
    Prediction,
    PreparedInput,
    RawOutput,
    Result,
    Step,
    StepRegistry,
)
from ai_inference.processing.decoders.sface import TARGET_POINTS

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ["yolo11", "rfdetr", "rtdetr", "yunet", "sface", "rfdetr_om"]


def cfg(family):
    if family in ("rfdetr", "rfdetr_om"):
        return load_synthetic_config(ROOT / "tests/fixtures" / f"{family}_synthetic.yaml")
    return load_synthetic_config(ROOT / "configs" / f"{family}.yaml")


def outputs(config, n):
    tensors = {
        s.name: np.zeros((n, *(d or 1 for d in s.shape[1:])), dtype=s.dtype) for s in config.outputs
    }
    if "fc1" in tensors:
        tensors["fc1"][:] = 1
    elif "output0" in tensors:
        tensors["output0"][:, :4, 0] = [320, 320, 40, 20]
        tensors["output0"][:, 4, 0] = 0.9
    elif "dets" in tensors:
        tensors["dets"][:] = 0.5
        tensors["labels"][:] = -100
        tensors["labels"][:, 0, 0] = 100
    elif "scores" in tensors:
        tensors["scores"][:, 0] = 0.9
        tensors["boxes"][:, 0] = [1, 2, 3, 4]
    return tensors


class PipelineEngine(BaseEngine):
    def __init__(self, config):
        self.config, self.calls = config, []

    def _infer(self, tensors):
        self.calls.append(len(next(iter(tensors.values()))))
        return outputs(self.config, self.calls[-1])


EngineRegistry.register("pipeline_test", PipelineEngine)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("n", [0, 1, 5])
def test_default_pipeline_batch_contracts(family, n, monkeypatch, caplog):
    current = cfg(family)
    random = np.random.default_rng(42)
    images = [random.integers(0, 256, (31 + i, 47 + i, 3), np.uint8) for i in range(n)]
    if family == "sface":
        images = [FaceInput(image, TARGET_POINTS) for image in images]
    data = current.model_dump()
    data["runtime"].update(backend="pipeline_test", warmup=False)
    model = Model(Config.model_validate(data))
    caplog.set_level(logging.INFO)
    actual_prepared = model.preprocess(images)
    assert actual_prepared.count == n
    assert current.inputs is not None
    for spec in current.inputs:
        assert spec.shape is not None
        tensor = actual_prepared.tensors[spec.name]
        assert tensor.shape == (n, *spec.shape[1:])
        assert tensor.dtype == np.dtype(spec.dtype) and tensor.flags.c_contiguous
    assert [meta.original_shape for meta in actual_prepared.metadata] == [
        (31 + i, 47 + i) for i in range(n)
    ]
    actual_raw = model.infer(actual_prepared)
    actual = model.postprocess(actual_raw)
    assert len(actual) == n
    if n and family == "sface":
        for result in actual:
            assert isinstance(result, FeaturePrediction)
            np.testing.assert_allclose(np.linalg.norm(result.embedding), 1.0, atol=1e-7)
    assert isinstance(model.preprocessor, Pipeline)
    assert not hasattr(model, "adapter")
    assert not any(hasattr(r, "preprocess_time") for r in caplog.records)
    physical = current.runtime.batch.size
    assert model.engine.calls == [physical] * ((n + physical - 1) // physical)


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda d: d["preprocess"]["steps"].insert(0, {"unknown": {}}), "Unknown"),
        (lambda d: d["preprocess"]["steps"][0].update(color={}), "exactly one"),
        (lambda d: d["preprocess"]["steps"][0]["resize"].update(witdth=640), "Extra inputs"),
        (lambda d: d["preprocess"]["steps"][0]["resize"].update(device="cuda"), "not implemented"),
        (lambda d: d["preprocess"].update(device="trt"), "not implemented"),
        (lambda d: d["preprocess"]["steps"].reverse(), "shape"),
        (lambda d: d["preprocess"]["steps"].pop(), "must end"),
        (lambda d: d["preprocess"]["steps"][2]["color"].update(source="RGB"), "source/layout"),
        (lambda d: d["preprocess"]["steps"][3]["normalize"].update(std=[1, 0, 1]), "positive"),
        (lambda d: d["preprocess"]["steps"][1]["padding"].update(width=320), "shape"),
        (lambda d: d["postprocess"]["steps"][0]["yolo_decode"].update(num_classes=2), "output"),
    ],
)
def test_invalid_plan_rejected_before_session(mutate, match, monkeypatch):
    data = cfg("yolo11").model_dump()
    mutate(data)

    def fail(*args):
        pytest.fail("invalid pipeline must not allocate an engine")

    monkeypatch.setattr(EngineRegistry, "create", fail)
    with pytest.raises((ValueError, TypeError), match=match):
        Model(Config.model_validate(data))


def test_step_cpu_override():
    data = cfg("yolo11").model_dump()
    data["runtime"].update(backend="pipeline_test", warmup=False)
    for entry in data["preprocess"]["steps"]:
        next(iter(entry.values()))["device"] = "cpu"
    assert len(Model(Config.model_validate(data)).predict(np.zeros((13, 19, 3), np.uint8))) == 1


def test_custom_registered_step_and_context_isolation():
    seen = []

    @StepRegistry.register("test_identity")
    def identity(config, **options):
        if options:
            raise ValueError("identity accepts no parameters")

        def run(context):
            assert isinstance(context, Context)
            seen.append(context)
            context.attributes["only_this_request"] = True

        return Step("images", "images", run, lambda state: None)

    data = cfg("yolo11").model_dump()
    data["preprocess"]["steps"].insert(0, {"test_identity": {}})
    pipeline = Pipeline("preprocess", data["preprocess"], Config.model_validate(data))
    pipeline([])
    pipeline([])
    assert seen[0] is not seen[1]
    assert seen[0].attributes is not seen[1].attributes
    with pytest.raises(ValueError, match="duplicate"):
        StepRegistry.register("test_identity", identity)


def test_layout_before_normalize_rejected():
    data = cfg("yolo11").model_dump()
    data["preprocess"]["steps"][3:5] = list(reversed(data["preprocess"]["steps"][3:5]))
    with pytest.raises(ValueError, match="HWC"):
        Pipeline("preprocess", data["preprocess"], Config.model_validate(data))


def test_source_rgb_and_model_tensor_name():
    data = cfg("yolo11").model_dump()
    data["preprocess"]["source_color"] = "RGB"
    data["preprocess"]["steps"][2]["color"]["source"] = "RGB"
    data["preprocess"]["steps"][-1]["pack"]["input_name"] = "renamed"
    data["inputs"][0]["name"] = "renamed"
    pipeline = Pipeline("preprocess", data["preprocess"], Config.model_validate(data))
    result = pipeline(np.full((640, 640, 3), [255, 0, 0], np.uint8))
    np.testing.assert_array_equal(result.tensors["renamed"][0, :, 0, 0], [1, 0, 0])


def test_pipeline_injection_preserves_final_contract():
    data = cfg("rfdetr").model_dump()
    data["runtime"].update(backend="pipeline_test", warmup=False)
    config = Config.model_validate(data)

    def pre(images):
        shape = (1, *data["inputs"][0]["shape"][1:])
        return PreparedInput({"input": np.zeros(shape, np.float32)}, ({"custom": True},))

    def post(raw) -> list[Result]:
        from ai_inference import Prediction

        assert raw.prepared.metadata == ({"custom": True},)
        return [
            Prediction(
                np.empty((0, 4), np.float32), np.empty(0, np.float32), np.empty(0, np.int64), (1, 1)
            )
        ]

    model = Model(config, preprocessor=pre, postprocessor=post)
    assert model.preprocessor is pre and model.postprocessor is post
    assert len(model.predict(np.zeros((1, 1, 3), np.uint8))) == 1


def test_mixed_pipeline_and_flat_parameters_rejected():
    data = cfg("rfdetr").model_dump()
    data["postprocess"] = {"num_classes": 2}
    with pytest.raises(ValueError, match="both use"):
        Config.model_validate(data)


def test_face_alignment_cannot_feed_detector():
    data = cfg("yolo11").model_dump()
    data["preprocess"]["steps"].insert(0, {"face_align": {}})
    config = Config.model_validate(data)
    pre = Pipeline("preprocess", config.preprocess, config)(
        FaceInput(np.zeros((112, 112, 3), np.uint8), TARGET_POINTS)
    )
    with pytest.raises(ValueError, match="face-aligned"):
        Pipeline("postprocess", config.postprocess, config)(RawOutput(outputs(config, 1), pre))


def test_composed_geometry_and_no_input_mutation():
    pipeline = Pipeline("preprocess", cfg("yolo11").preprocess, cfg("yolo11"))
    image = np.full((101, 200, 3), [1, 2, 3], np.uint8)
    untouched = image.copy()
    prepared = pipeline(image)
    meta = prepared.metadata[0]
    assert meta.original_shape == (101, 200)
    assert meta.scale == (3.2, 323 / 101)
    assert meta.gain == 3.2 and meta.padding == (0, 158)
    np.testing.assert_array_equal(image, untouched)


def test_filter_preserves_instance_alignment():
    from ai_inference import Prediction
    from ai_inference.processing.steps.postprocess import filter_detections

    result = Prediction(
        np.arange(8).reshape(2, 4),
        np.array([0.3, 0.9]),
        np.array([0, 1]),
        (4, 5),
        keypoints=np.arange(20).reshape(2, 5, 2),
        masks=np.ones((2, 4, 5)),
    )
    context = Context([result], "results")
    filter_detections(cfg("yolo11"), class_ids=[1], threshold=0.5).run(context)
    [filtered] = context.data
    for field in ("boxes", "scores", "class_ids", "keypoints", "masks"):
        np.testing.assert_array_equal(getattr(filtered, field), getattr(result, field)[1:])


def test_result_steps_compose_in_order():
    data = cfg("yolo11").model_dump()
    data["postprocess"]["steps"].append({"filter_detections": {"threshold": 0.95}})
    data["runtime"].update(backend="pipeline_test", warmup=False)
    result = Model(Config.model_validate(data)).predict(np.zeros((13, 19, 3), np.uint8))[0]
    assert isinstance(result, Prediction)
    assert len(result.boxes) == 0


def test_pipeline_request_contexts_are_concurrent_safe():
    from concurrent.futures import ThreadPoolExecutor

    config = cfg("rfdetr")
    pipeline = Pipeline("preprocess", config.preprocess, config)
    images = [np.zeros((31 + i, 47 + i, 3), np.uint8) for i in range(8)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(pipeline, images))
    shapes = []
    for result in results:
        assert isinstance(result, PreparedInput)
        shapes.append(result.metadata[0].original_shape)
    assert shapes == [i.shape[:2] for i in images]


def test_scale_and_divisor_are_explicit_alternatives():
    data = cfg("yolo11").model_dump()
    normalize = data["preprocess"]["steps"][3]["normalize"]
    normalize.pop("divisor")
    normalize["scale"] = 1 / 255
    config = Config.model_validate(data)
    image = np.full((640, 640, 3), [1, 2, 3], np.uint8)
    actual = Pipeline("preprocess", config.preprocess, config)(image)
    np.testing.assert_allclose(
        actual.tensors["images"][0, :, 0, 0], np.array([3, 2, 1]) / 255, atol=1e-9
    )
    normalize["divisor"] = 255
    with pytest.raises(ValueError, match="not both"):
        Pipeline("preprocess", data["preprocess"], Config.model_validate(data))


def test_connection_contract_rejected_before_session(monkeypatch):
    data = cfg("yolo11").model_dump()
    data["preprocess"]["steps"][2]["color"]["target"] = "BGR"

    def fail(*args):
        pytest.fail("invalid color must be rejected before session creation")

    monkeypatch.setattr(EngineRegistry, "create", fail)
    with pytest.raises(ValueError, match="color must be RGB"):
        Model(Config.model_validate(data))
    data = cfg("yolo11").model_dump()
    data["preprocess"]["steps"][0]["resize"]["keep_ratio"] = False
    with pytest.raises(ValueError, match="uniform"):
        Model(Config.model_validate(data))


def test_custom_step_runtime_output_contract():
    @StepRegistry.register("test_wrong_kind")
    def broken(config, **options):
        def run(context):
            context.kind = "results"

        return Step("images", "images", run, lambda state: None)

    data = cfg("rfdetr").model_dump()
    data["preprocess"]["steps"].insert(0, {"test_wrong_kind": {}})
    config = Config.model_validate(data)
    with pytest.raises(ValueError, match="violated"):
        Pipeline("preprocess", config.preprocess, config)([])


def test_dynamic_pipeline_split_and_empty():
    data = cfg("rfdetr").model_dump()
    data["runtime"].update(backend="pipeline_test", warmup=False)
    data["runtime"]["batch"].update(mode="dynamic", size=2)
    for spec in data["inputs"] + data["outputs"]:
        spec["shape"] = (None, *spec["shape"][1:])
    model = Model(Config.model_validate(data))
    assert len(model.predict([np.zeros((13, 19, 3), np.uint8)] * 5)) == 5
    assert model.engine.calls == [2, 2, 1]
    assert model.predict([]) == []


@pytest.mark.parametrize(
    "change,match",
    [
        (lambda d: d["model"].update(adapter="rfdetr"), "Extra inputs"),
        (lambda d: d.update(processing={"preprocess": {"device": "cpu"}}), "Extra inputs"),
        (
            lambda d: d.update(preprocess={"height": 560}, postprocess={"num_classes": 2}),
            "ordered steps",
        ),
    ],
)
def test_removed_configuration_rejected(change, match):
    data = cfg("rfdetr").model_dump()
    change(data)
    with pytest.raises(ValueError, match=match):
        Config.model_validate(data)


def test_yunet_padding_restores_boxes_and_landmarks():
    data = cfg("yunet").model_dump()
    data["preprocess"]["steps"][0] = {"resize": {"width": 320, "height": 320}}
    data["preprocess"]["steps"].insert(1, {"padding": {"width": 640, "height": 640, "value": 0}})
    config = Config.model_validate(data)
    pre = Pipeline("preprocess", config.preprocess, config)(np.zeros((320, 320, 3), np.uint8))
    tensors = outputs(config, 1)
    # Grid cell (24,24) on stride 8: center (192,192), 16x16 box.
    index = 24 * 80 + 24
    tensors["cls_8"][0, index, 0] = 1
    tensors["obj_8"][0, index, 0] = 1
    tensors["bbox_8"][0, index] = [0, 0, np.log(2), np.log(2)]
    result = Pipeline("postprocess", config.postprocess, config)(RawOutput(tensors, pre))[0]
    assert isinstance(result, Prediction) and result.keypoints is not None
    np.testing.assert_allclose(result.boxes, [[24, 24, 40, 40]], atol=1e-6)
    np.testing.assert_allclose(result.keypoints, [[[32, 32]] * 5], atol=1e-6)


def test_sface_normalization_is_an_independent_result_step():
    data = cfg("sface").model_dump()
    data["postprocess"]["steps"].pop()
    config = Config.model_validate(data)
    prepared = Pipeline("preprocess", config.preprocess, config)(
        FaceInput(np.zeros((112, 112, 3), np.uint8), TARGET_POINTS)
    )
    tensors = {"fc1": np.zeros((1, 128), np.float32)}
    tensors["fc1"][0, :2] = [3, 4]
    result = Pipeline("postprocess", config.postprocess, config)(RawOutput(tensors, prepared))[0]
    assert isinstance(result, FeaturePrediction)
    np.testing.assert_array_equal(result.embedding[:2], [3, 4])
    data["postprocess"]["steps"].append({"l2_normalize": {}})
    config = Config.model_validate(data)
    result = Pipeline("postprocess", config.postprocess, config)(RawOutput(tensors, prepared))[0]
    assert isinstance(result, FeaturePrediction)
    np.testing.assert_allclose(result.embedding[:2], [0.6, 0.8])
