import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from loguru import logger as loguru_logger
from model_contracts import load_synthetic_config
from test_ort import fake_ort

from ai_inference import Config, Model, ONNXRuntimeEngine, PreparedInput
from ai_inference.inference.predictor import DefaultPredictor


@pytest.mark.parametrize("constraints", ["omitted", "names", "dtype", "shape"])
def test_discovery_and_model_logger(monkeypatch, tmp_path, config, caplog, constraints):
    explicit, _ = fake_ort(monkeypatch, tmp_path, config)
    data = explicit.model_dump()
    for kind in ("inputs", "outputs"):
        if constraints == "omitted":
            data.pop(kind)
        else:
            data[kind] = [
                {key: value for key, value in spec.items() if key == "name" or key == constraints}
                for spec in data[kind]
            ]
    logger = loguru_logger.bind(application="test.discovery")
    with caplog.at_level(logging.INFO):
        model = Model(Config.model_validate(data), logger=logger)
    assert model.config.inputs == explicit.inputs
    assert model.config.outputs == explicit.outputs
    assert model.engine.config == model.config
    assert model.predictor.config == model.config
    records = [
        record
        for record in caplog.records
        if getattr(record, "application", None) == "test.discovery"
        and getattr(record, "step", None) == "engine.metadata"
    ]
    assert len(records) == 3
    assert "[rfdetr_synthetic]-[init]-[engine.metadata]" in records[0].message
    assert "Input 'input' | float32 | shape=[4, 3, 560, 560]" in records[0].message
    assert model.preprocess(np.zeros((20, 30, 3), np.uint8)).count == 1


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("name", "wrong", "names"),
        ("dtype", "int64", "dtype"),
        ("shape", (4, 3, 320, 320), "shape"),
    ],
)
def test_optional_constraints_reject_conflicts(monkeypatch, tmp_path, config, field, value, match):
    explicit, _ = fake_ort(monkeypatch, tmp_path, config)
    data = explicit.model_dump()
    data.pop("outputs")
    data["inputs"] = [{"name": "input", field: value}]
    with pytest.raises(ValueError, match=match):
        ONNXRuntimeEngine(Config.model_validate(data))


@pytest.mark.parametrize("mode,batch", [("fixed", 4), ("fixed", "batch"), ("dynamic", "batch")])
def test_real_ort_discovered_batch_contract(tmp_path, config, mode, batch):
    import onnx
    from onnx import TensorProto, helper

    graph = helper.make_graph(
        [helper.make_node("Identity", ["x"], ["y"])],
        "identity",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [batch, 2])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [batch, 2])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    path = tmp_path / "identity.onnx"
    onnx.save(model, path)
    data = config.model_dump()
    data["model"]["path"] = path
    data["runtime"]["batch"]["mode"] = mode
    data["runtime"]["options"]["providers"] = "cpu"
    data.pop("inputs")
    data.pop("outputs")
    engine = ONNXRuntimeEngine(Config.model_validate(data))
    assert engine.config.inputs is not None
    assert engine.config.inputs[0].shape == (4 if mode == "fixed" else None, 2)
    assert engine.read_tensor_metadata()["inputs"][0].shape == (batch, 2)
    for count in (1, 3, 4, 5):
        x = np.arange(count * 2, dtype=np.float32).reshape(count, 2)
        prepared = PreparedInput({"x": x}, tuple(range(count)))
        np.testing.assert_array_equal(
            DefaultPredictor(engine, engine.config)(prepared).tensors["y"], x
        )


@pytest.mark.parametrize("mode", ["fixed", "dynamic"])
@pytest.mark.parametrize("failure", [None, "box_width", "candidate_counts"])
def test_rtdetr_symbolic_outputs_checked_at_runtime(monkeypatch, tmp_path, mode, failure):
    config = load_synthetic_config(Path(__file__).resolve().parents[1] / "configs/rtdetr.yaml")
    data = config.model_dump()
    data["runtime"].update(warmup=False)
    data["runtime"]["batch"].update(mode=mode, size=2)
    for kind in ("inputs", "outputs"):
        for spec in data[kind]:
            spec["shape"] = (2 if mode == "fixed" else None, *spec["shape"][1:])
    explicit, _ = fake_ort(
        monkeypatch,
        tmp_path,
        Config.model_validate(data),
        available=["CPUExecutionProvider"],
        active=["CPUExecutionProvider"],
    )
    import sys

    session = sys.modules["onnxruntime"].InferenceSession
    session.get_inputs = lambda self: [
        SimpleNamespace(
            name=s.name,
            type=f"tensor({'int64' if s.dtype == 'int64' else 'float'})",
            shape=["N", *s.shape[1:]],
        )
        for s in explicit.inputs
    ]
    session.get_outputs = lambda self: [
        SimpleNamespace(name="labels", type="tensor(int64)", shape=["N", "K"]),
        SimpleNamespace(name="boxes", type="tensor(float)", shape=["N", "K", "coords"]),
        SimpleNamespace(name="scores", type="tensor(float)", shape=["N", "K"]),
    ]

    def run(self, names, feeds):
        n = len(feeds["images"])
        values = {
            "labels": np.zeros((n, 2), np.int64),
            "boxes": np.zeros((n, 2, 3 if failure == "box_width" else 4), np.float32),
            "scores": np.ones((n, 3 if failure == "candidate_counts" else 2), np.float32),
        }
        return [values[name] for name in names]

    session.run = run
    data = explicit.model_dump()
    data.pop("inputs")
    data.pop("outputs")
    model = Model(Config.model_validate(data))
    assert model.config.outputs[1].shape == (2 if mode == "fixed" else None, None, None)
    image = np.zeros((20, 30, 3), np.uint8)
    if failure:
        with pytest.raises(ValueError, match="inconsistent RT-DETR output counts"):
            model.predict(image)
    else:
        results = model.predict([image] * 5)
        assert len(results) == 5 and all(r.class_ids.tolist() == [0, 0] for r in results)
        assert model.predict([]) == []


@pytest.mark.parametrize("kind", ["inputs", "outputs"])
@pytest.mark.parametrize("mode", ["fixed", "dynamic"])
def test_discovered_static_batch_policy_mismatch(monkeypatch, tmp_path, config, kind, mode):
    import sys

    explicit, _ = fake_ort(monkeypatch, tmp_path, config)
    session = sys.modules["onnxruntime"].InferenceSession
    for direction in ("inputs", "outputs"):
        method = f"get_{direction}"
        original = getattr(session, method)

        def metadata(self, original=original, direction=direction):
            nodes = original(self)
            for node in nodes:
                node.shape = [2 if direction == kind else "N", *node.shape[1:]]
            return nodes

        monkeypatch.setattr(session, method, metadata)
    data = explicit.model_dump()
    data.pop("inputs")
    data.pop("outputs")
    data["runtime"]["batch"].update(mode=mode, size=4)
    with pytest.raises(ValueError) as caught:
        ONNXRuntimeEngine(Config.model_validate(data))
    message = str(caught.value)
    assert f"{kind} tensor '" in message
    assert "model batch=2" in message
    assert f"runtime.batch.mode={mode}, size=4" in message
    assert "use mode=fixed, size=2" in message


@pytest.mark.parametrize(
    "field,value", [("shape", (1, 300, 3)), ("shape", (1, 299, 4)), ("dtype", "int64")]
)
def test_rtdetr_known_output_conflicts_rejected(field, value):
    from pipeline_helpers import pipelines

    config = load_synthetic_config(Path(__file__).resolve().parents[1] / "configs/rtdetr.yaml")
    data = config.model_dump()
    data["outputs"][1][field] = value
    with pytest.raises(ValueError):
        pipelines(Config.model_validate(data))


def test_real_ort_model_discovery_predict_and_warmup(tmp_path, config, caplog):
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    nodes, outputs = [], []
    for spec in config.outputs:
        values = np.zeros(spec.shape, np.float32)
        nodes.append(
            helper.make_node("Constant", [], [spec.name], value=numpy_helper.from_array(values))
        )
        outputs.append(
            helper.make_tensor_value_info(spec.name, TensorProto.FLOAT, list(spec.shape))
        )
    graph = helper.make_graph(
        nodes,
        "detection",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [4, 3, 560, 560])],
        outputs,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    path = tmp_path / "detection.onnx"
    onnx.save(model, path)
    data = config.model_dump()
    data["model"]["path"] = path
    data["runtime"].update(warmup=True, warmup_runs=1)
    data["runtime"]["options"]["providers"] = "cpu"
    data.pop("inputs")
    data.pop("outputs")
    with caplog.at_level(logging.INFO, logger="ai_inference.model"):
        sdk = Model(Config.model_validate(data))
    assert "shape=[4, 3, 560, 560]" in caplog.text
    assert len(sdk.predict(np.zeros((20, 30, 3), np.uint8))) == 1
    assert len(sdk.predict([np.zeros((20, 30, 3), np.uint8)] * 5)) == 5


def test_other_backends_require_explicit_contract(config):
    data = config.model_dump()
    data["runtime"]["backend"] = "custom_explicit_backend"
    data.pop("inputs")
    with pytest.raises(ValueError, match="requires explicit"):
        Config.model_validate(data)


def test_dynamic_model_logs_original_symbolic_shape(monkeypatch, tmp_path, config, caplog):
    import sys

    data = config.model_dump()
    data["runtime"]["batch"]["mode"] = "dynamic"
    for kind in ("inputs", "outputs"):
        for spec in data[kind]:
            spec["shape"] = (None, *spec["shape"][1:])
    explicit, _ = fake_ort(monkeypatch, tmp_path, Config.model_validate(data))
    session = sys.modules["onnxruntime"].InferenceSession
    for method in ("get_inputs", "get_outputs"):
        original = getattr(session, method)

        def symbolic(self, original=original):
            nodes = original(self)
            for node in nodes:
                node.shape = ["batch", *node.shape[1:]]
            return nodes

        monkeypatch.setattr(session, method, symbolic)
    data = explicit.model_dump()
    data.pop("inputs")
    data.pop("outputs")
    with caplog.at_level(logging.INFO, logger="ai_inference.model"):
        model = Model(Config.model_validate(data))
    assert "shape=['batch', 3, 560, 560]" in caplog.text
    assert model.config.inputs is not None
    assert model.config.inputs[0].shape == (None, 3, 560, 560)
