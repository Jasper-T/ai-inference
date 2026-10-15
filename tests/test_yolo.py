from pathlib import Path

import numpy as np
import pytest
from model_contracts import load_synthetic_config
from pipeline_helpers import options, pipelines

from ai_inference import Config, Model, Prediction, RawOutput


@pytest.fixture
def yolo_config():
    data = load_synthetic_config(
        Path(__file__).resolve().parents[1] / "configs/yolo11.yaml"
    ).model_dump()
    data["outputs"][0]["shape"] = (1, 6, 4)
    options(data, "yolo_decode")["num_classes"] = 2
    return Config.model_validate(data)


def test_preprocess(yolo_config):
    pair = pipelines(yolo_config)
    image = np.full((100, 200, 3), [0, 127, 255], np.uint8)
    prepared = pair.preprocess(image)
    tensor = prepared.tensors["images"]
    assert tensor.shape == (1, 3, 640, 640) and tensor.flags.c_contiguous
    np.testing.assert_allclose(tensor[0, :, 160, 0], [1, 127 / 255, 0], atol=1e-7)
    np.testing.assert_allclose(tensor[0, :, 159, 0], 114 / 255, atol=1e-7)
    assert prepared.metadata[0].padding == (0, 160)
    assert pair.preprocess([]).count == 0
    with pytest.raises(ValueError):
        pair.preprocess(image.astype(np.float32))


def raw(pair):
    prepared = pair.preprocess(np.zeros((100, 200, 3), np.uint8))
    # First two overlap in same class; third overlaps but belongs to a different class.
    rows = np.array(
        [
            [320, 320, 320, 160, 0.9, 0.1],
            [320, 320, 320, 160, 0.8, 0.1],
            [320, 320, 320, 160, 0.1, 0.85],
            [50, 50, 10, 10, 0.1, 0.1],
        ],
        np.float32,
    )
    values = rows.T[None]
    if options(pair.config, "yolo_decode").get("layout", "NCQ") == "NQC":
        values = values.transpose(0, 2, 1)
    values = values.astype(pair.config.outputs[0].dtype)
    return RawOutput({"output0": values}, prepared)


def test_nms_inverse_and_no_sigmoid(yolo_config):
    pair = pipelines(yolo_config)
    result = pair.postprocess(raw(pair))[0]
    assert result.class_ids.tolist() == [0, 1]
    np.testing.assert_allclose(result.scores, [0.9, 0.85])
    np.testing.assert_allclose(result.boxes, [[50, 25, 150, 75]] * 2)


@pytest.mark.parametrize(
    "changes,expected",
    [
        ({"agnostic_nms": True}, [0]),
        ({"max_detections": 1}, [0]),
        ({"confidence_threshold": 0.95}, []),
    ],
)
def test_filters(yolo_config, changes, expected):
    data = yolo_config.model_dump()
    options(data, "yolo_decode").update(changes)
    pair = pipelines(Config.model_validate(data))
    result = pair.postprocess(raw(pair))[0]
    assert result.class_ids.tolist() == expected
    assert result.boxes.shape == (len(expected), 4)


def test_fp16_transposed_and_validation(yolo_config):
    data = yolo_config.model_dump()
    options(data, "normalize")["dtype"] = "float16"
    data["inputs"][0]["dtype"] = "float16"
    data["outputs"][0].update(dtype="float16", shape=(1, 4, 6))
    options(data, "yolo_decode")["layout"] = "NQC"
    pair = pipelines(Config.model_validate(data))
    output = raw(pair)
    assert output.prepared.tensors["images"].dtype == np.float16
    assert pair.postprocess(output)[0].class_ids.tolist() == [0, 1]
    output.tensors["output0"][0, 0, 4] = 2
    with pytest.raises(ValueError, match="probabilities"):
        pair.postprocess(output)
    output.tensors["output0"][0, 0, 4] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        pair.postprocess(output)
    data["outputs"][0]["shape"] = (1, 4, 7)
    with pytest.raises(ValueError, match="YOLO output"):
        pipelines(Config.model_validate(data))


def test_real_ort_pipeline(tmp_path, yolo_config):
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    pair = pipelines(yolo_config)
    value = raw(pair).tensors["output0"]
    graph = helper.make_graph(
        [helper.make_node("Constant", [], ["output0"], value=numpy_helper.from_array(value))],
        "yolo_contract",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 640, 640])],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 6, 4])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    path = tmp_path / "yolo.onnx"
    onnx.save(model, path)
    data = yolo_config.model_dump()
    data["model"]["path"] = path
    data["runtime"]["options"].update(providers="cpu")
    sdk = Model(Config.model_validate(data))
    results = sdk.predict([np.zeros((100, 200, 3), np.uint8), np.zeros((200, 400, 3), np.uint8)])
    assert len(results) == 2
    assert isinstance(results[0], Prediction) and isinstance(results[1], Prediction)
    np.testing.assert_allclose(results[1].boxes, results[0].boxes * 2)
    assert sdk.predict([]) == []


@pytest.mark.parametrize("threshold", [0.0, 0.25, 1.0])
def test_confidence_inclusive(yolo_config, threshold):
    data = yolo_config.model_dump()
    options(data, "yolo_decode")["confidence_threshold"] = threshold
    pair = pipelines(Config.model_validate(data))
    output = raw(pair)
    values = output.tensors["output0"]
    values[:] = 0
    values[0, :4, 0] = [100, 100, 20, 20]
    values[0, 4, 0] = threshold
    if threshold > 0:
        values[0, :4, 1] = [300, 300, 20, 20]
        values[0, 4, 1] = np.nextafter(np.float32(threshold), np.float32(0))
    result = pair.postprocess(output)[0]
    assert len(result.scores) == 1
    assert result.scores[0] == threshold


@pytest.mark.parametrize("explicit", [False, True])
def test_output_name_resolution(yolo_config, explicit):
    data = yolo_config.model_dump()
    data["outputs"][0]["name"] = "predictions"
    options(data, "yolo_decode").pop("output_name", None)
    if explicit:
        options(data, "yolo_decode")["output_name"] = "predictions"
    config = Config.model_validate(data)
    pair = pipelines(config)
    output = raw(pair)
    renamed = RawOutput({"predictions": output.tensors["output0"]}, output.prepared)
    assert pair.config.outputs[0].name == "predictions"
    assert pair.postprocess(renamed)[0].class_ids.tolist() == [0, 1]


def test_explicit_output_mismatch(yolo_config):
    data = yolo_config.model_dump()
    options(data, "yolo_decode")["output_name"] = "wrong"
    config = Config.model_validate(data)
    for factory in [pipelines]:
        with pytest.raises(ValueError, match="output_name"):
            factory(config)
