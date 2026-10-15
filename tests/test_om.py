import sys
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace

import cv2
import numpy as np
import pytest
from model_contracts import load_synthetic_config
from pipeline_helpers import options, pipelines

from ai_inference import Config, Model, OMEngine, RawOutput


@pytest.fixture
def om_config():
    return load_synthetic_config(
        Path(__file__).resolve().parents[1] / "tests/fixtures/rfdetr_om_synthetic.yaml"
    )


def install_session(monkeypatch, factory, metadata_config=None):
    for name in ["ais_bench", "ais_bench.infer", "ais_bench.infer.interface"]:
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    if isinstance(factory, type):
        config = metadata_config or load_synthetic_config(
            Path(__file__).resolve().parents[1] / "tests/fixtures/rfdetr_om_synthetic.yaml"
        )
        dtype = Enum("dtype", "float32 float16 double64 int64 int32 uint8 bool int8")

        def nodes(kind):
            return [
                SimpleNamespace(
                    name=s.name,
                    datatype=dtype["double64" if s.dtype == "float64" else s.dtype],
                    shape=list(s.shape),
                )
                for s in getattr(config, kind)
            ]

        factory.get_inputs = lambda self: nodes("inputs")
        factory.get_outputs = lambda self: nodes("outputs")
    setattr(sys.modules["ais_bench.infer.interface"], "InferSession", factory)


def test_aipp_pipeline_pixels(om_config):
    pair = pipelines(om_config)
    image = np.full((100, 200, 3), [11, 22, 33], np.uint8)
    prep = pair.preprocess(image)
    np.testing.assert_array_equal(prep.tensors["input"][0], cv2.resize(image, (560, 560)))
    assert prep.tensors["input"].dtype == np.uint8 and prep.tensors["input"].flags.c_contiguous
    assert pair.preprocess([]).tensors["input"].shape == (0, 560, 560, 3)
    boxes = np.full((1, 300, 4), 0.5, np.float32)
    logits = np.full((1, 300, 2), -100, np.float32)
    logits[0, 0, 0] = 100
    result = pair.postprocess(RawOutput({"dets": boxes, "labels": logits}, prep))[0]
    np.testing.assert_array_equal(result.boxes, [[50, 25, 150, 75]])
    assert result.class_ids.tolist() == [0]


def test_official_nano_om_example_contract_and_decode_match_onnx(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    original = Config.from_yaml(root / "configs/rfdetr_om.yaml")
    metadata = original.model_dump()
    metadata["inputs"] = [dict(name="input", dtype="uint8", shape=(1, 384, 384, 3))]
    metadata["outputs"] = [
        dict(name="dets", dtype="float32", shape=(1, 300, 4)),
        dict(name="labels", dtype="float32", shape=(1, 300, 91)),
    ]
    contract = Config.model_validate(metadata)
    boxes = np.full((1, 300, 4), 0.5, np.float32)
    logits = np.full((1, 300, 91), -100, np.float32)
    logits[0, 0, 90] = 100

    class Session:
        def __init__(self, device, path):
            assert Path(path).name == "rfdetr-nano.om"

        def infer(self, feeds):
            assert feeds[0].shape == (1, 384, 384, 3)
            assert feeds[0].dtype == np.uint8
            return [boxes, logits]

    install_session(monkeypatch, Session, contract)
    image = np.zeros((73, 119, 3), np.uint8)
    actual = Model(original).predict(image)[0]
    onnx = Config.from_yaml(root / "configs/rfdetr.yaml").model_dump()
    onnx["inputs"] = [dict(name="input", dtype="float32", shape=(1, 3, 384, 384))]
    onnx["outputs"] = metadata["outputs"]
    reference = pipelines(Config.model_validate(onnx))
    expected = reference.postprocess(
        RawOutput({"dets": boxes, "labels": logits}, reference.preprocess(image))
    )[0]
    assert actual.class_ids.tolist() == [90]
    np.testing.assert_array_equal(actual.boxes, expected.boxes)
    np.testing.assert_array_equal(actual.scores, expected.scores)


def test_om_warmup_and_batch(monkeypatch, om_config):
    calls = []

    class Session:
        def __init__(self, device, path):
            assert device == 0 and path == str(om_config.model.path)

        def infer(self, feeds):
            calls.append(feeds[0].copy())
            return [np.zeros((4, 300, 4), np.float32), np.full((4, 300, 2), -100, np.float32)]

    install_session(monkeypatch, Session)
    model = Model(om_config)
    assert len(calls) == 3 and all(not x.any() for x in calls)
    assert len(model.predict([np.ones((5, 8, 3), np.uint8)] * 5)) == 5
    assert len(calls) == 5 and all(x.shape == (4, 560, 560, 3) for x in calls)
    assert model.predict([]) == [] and len(calls) == 5


def test_order_and_execution_validation(monkeypatch, om_config):
    data = om_config.model_dump()
    data["inputs"] = [dict(name=n, dtype="float32", shape=(4, 2)) for n in ["b", "a"]]
    data["outputs"] = [dict(name=n, dtype="float32", shape=(4, 2)) for n in ["second", "first"]]
    config = Config.model_validate(data)

    class Session:
        def __init__(self, *args):
            pass

        def infer(self, feeds):
            assert all(x.flags.c_contiguous for x in feeds)
            np.testing.assert_array_equal(feeds[0], 2)
            np.testing.assert_array_equal(feeds[1], 1)
            return feeds

    install_session(monkeypatch, Session, config)
    # YAML order must not change vendor positional feeds or output binding.
    data["inputs"].reverse()
    data["outputs"].reverse()
    engine = OMEngine(Config.model_validate(data))
    assert engine.config.inputs == config.inputs
    assert engine.config.outputs == config.outputs
    a = np.ones((4, 4), np.float32)[:, ::2]
    result = engine.infer({"a": a, "b": a * 2})
    np.testing.assert_array_equal(result["second"], 2)
    with pytest.raises(ValueError):
        engine.infer({"a": a})
    engine.session.infer = lambda feeds: []
    with pytest.raises(ValueError, match="count"):
        engine.infer({"a": a, "b": a * 2})
    engine.session.infer = lambda feeds: [a.astype(np.float16), a]
    with pytest.raises(ValueError):
        engine.infer({"a": a, "b": a * 2})


def test_error_passthrough(monkeypatch, om_config):
    error = RuntimeError("vendor failure")

    def fail(*args):
        raise error

    install_session(monkeypatch, fail)
    with pytest.raises(RuntimeError) as caught:
        OMEngine(om_config)
    assert caught.value is error


@pytest.mark.parametrize("constraints", ["omitted", "names", "dtype", "shape", "inputs", "outputs"])
def test_om_discovery_predict_and_warmup(monkeypatch, om_config, constraints):
    data = om_config.model_dump()
    # The same model can be exported with names different from the example.
    data["inputs"][0]["name"] = "images"
    data["outputs"][0]["name"] = "boxes"
    data["outputs"][1]["name"] = "logits"
    options(data, "pack")["input_name"] = "images"
    options(data, "rfdetr_decode").update(boxes_name="boxes", logits_name="logits")
    explicit = Config.model_validate(data)
    calls = []

    class Session:
        def __init__(self, device, path):
            pass

        def infer(self, feeds):
            calls.append(feeds[0].copy())
            return [np.zeros((4, 300, 4), np.float32), np.full((4, 300, 2), -100, np.float32)]

    install_session(monkeypatch, Session, explicit)
    for kind in ("inputs", "outputs"):
        if constraints == "omitted" or constraints == kind:
            data.pop(kind)
        elif constraints not in ("inputs", "outputs"):
            data[kind] = [
                {k: v for k, v in spec.items() if k == "name" or k == constraints}
                for spec in data[kind]
            ]
    original = Config.model_validate(data)
    snapshot = original.model_dump()
    model = Model(original)
    assert model.config.inputs == explicit.inputs
    assert model.config.outputs == explicit.outputs
    assert model.config == model.engine.config == model.predictor.config
    assert original.model_dump() == snapshot
    assert len(calls) == 3 and all(not feed.any() for feed in calls)
    assert len(model.predict([np.ones((5, 8, 3), np.uint8)] * 5)) == 5
    assert len(calls) == 5
    assert [m.name for m in model.engine.read_tensor_metadata()["outputs"]] == ["boxes", "logits"]


@pytest.mark.parametrize(
    "kind,field,value,match",
    [
        ("inputs", "name", "alias", "names"),
        ("outputs", "name", "alias", "names"),
        ("inputs", "dtype", "float32", "dtype"),
        ("outputs", "dtype", "float16", "dtype"),
        ("inputs", "shape", (4, 320, 320, 3), "shape"),
        ("outputs", "shape", (4, 300, 4, 1), "rank"),
    ],
)
def test_om_constraints_reject_conflicts(monkeypatch, om_config, kind, field, value, match):
    class Session:
        def __init__(self, *args):
            pass

        def infer(self, feeds):
            pytest.fail("contract mismatch must fail before execution")

    install_session(monkeypatch, Session, om_config)
    data = om_config.model_dump()
    data[kind][0][field] = value
    with pytest.raises(ValueError, match=match):
        OMEngine(Config.model_validate(data))


@pytest.mark.parametrize(
    "dtype", ["float32", "float16", "float64", "int64", "int32", "uint8", "bool"]
)
def test_om_metadata_dtype_and_execution(monkeypatch, om_config, dtype):
    data = om_config.model_dump()
    data["runtime"]["options"]["device_id"] = 2
    data["inputs"] = [{"name": "x", "dtype": dtype, "shape": (4, 2)}]
    data["outputs"] = [{"name": "y", "dtype": dtype, "shape": (4, 2)}]
    explicit = Config.model_validate(data)

    class Session:
        def __init__(self, device, path):
            assert device == 2

        def infer(self, feeds):
            return feeds

    install_session(monkeypatch, Session, explicit)
    data.pop("inputs")
    data.pop("outputs")
    engine = OMEngine(Config.model_validate(data))
    assert engine.config.inputs == explicit.inputs
    assert engine.config.outputs == explicit.outputs
    tensor = np.ones((4, 2), dtype=dtype)
    np.testing.assert_array_equal(engine.infer({"x": tensor})["y"], tensor)
    error = RuntimeError("vendor execution failure")

    def fail(feeds):
        raise error

    engine.session.infer = fail
    with pytest.raises(RuntimeError) as caught:
        engine.infer({"x": tensor})
    assert caught.value is error


@pytest.mark.parametrize("kind", ["inputs", "outputs"])
@pytest.mark.parametrize("invalid", ["dtype", "dynamic", "duplicate", "batch"])
def test_om_rejects_invalid_model_metadata(monkeypatch, om_config, kind, invalid):
    class Session:
        def __init__(self, *args):
            pass

    install_session(monkeypatch, Session, om_config)
    method = f"get_{kind}"
    original = getattr(Session, method)

    def invalid_nodes(self):
        nodes = original(self)
        if invalid == "dtype":
            nodes[0].datatype = SimpleNamespace(name="int8")
        elif invalid == "dynamic":
            nodes[0].shape[0] = -1
        elif invalid == "batch":
            nodes[0].shape[0] = 1
        else:
            nodes.append(nodes[0])
        return nodes

    monkeypatch.setattr(Session, method, invalid_nodes)
    data = om_config.model_dump()
    data.pop("inputs")
    data.pop("outputs")
    with pytest.raises(ValueError):
        OMEngine(Config.model_validate(data))


@pytest.mark.parametrize(
    "runtime",
    [
        {"options": {"device_id": -1}},
        {"options": {"providers": "cpu"}},
        {"batch": {"mode": "dynamic"}},
    ],
)
def test_om_invalid_runtime_rejected_before_session(monkeypatch, om_config, runtime):
    def fail(*args):
        pytest.fail("invalid runtime must not allocate a session")

    install_session(monkeypatch, fail)
    data = om_config.model_dump()
    data.pop("inputs")
    data.pop("outputs")
    data["runtime"].update(runtime)
    with pytest.raises(ValueError):
        OMEngine(Config.model_validate(data))


def test_aipp_reject_wrong_contract(om_config):
    data = om_config.model_dump()
    data["inputs"][0]["dtype"] = "float32"
    with pytest.raises(ValueError):
        pipelines(Config.model_validate(data))
    data = om_config.model_dump()
    options(data, "resize")["mean"] = [0, 0, 0]
    with pytest.raises(ValueError):
        pipelines(Config.model_validate(data))
