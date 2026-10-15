import cv2
import numpy as np
import pytest
from pipeline_helpers import options, pipelines

from ai_inference import Config, RawOutput


def test_uniform_threshold_filters_all_classes_and_includes_boundary(config):
    data = config.model_dump()
    options(data, "rfdetr_decode")["threshold"] = 0.5
    pair = pipelines(Config.model_validate(data))
    prepared = pair.preprocess(np.zeros((10, 10, 3), np.uint8))
    boxes = np.zeros((1, 300, 4), np.float32)
    logits = np.full((1, 300, 2), -100, np.float32)
    # Both classes: below, exactly at, and above the shared threshold.
    logits[0, :6] = [[-1, -100], [-100, -1], [0, -100], [-100, 0], [1, -100], [-100, 1]]
    result = pair.postprocess(RawOutput({"dets": boxes, "labels": logits}, prepared))[0]
    assert result.class_ids.tolist() == [0, 1, 0, 1]
    np.testing.assert_allclose(
        result.scores, [0.5, 0.5, 1 / (1 + np.exp(-1)), 1 / (1 + np.exp(-1))]
    )


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan"), float("inf"), [0.4, 0.4]])
def test_invalid_uniform_threshold(config, threshold):
    data = config.model_dump()
    options(data, "rfdetr_decode")["threshold"] = threshold
    with pytest.raises(ValueError):
        pipelines(Config.model_validate(data))


def test_per_class_thresholds_are_rejected(config):
    data = config.model_dump()
    options(data, "rfdetr_decode")["thresholds"] = [0.4, 0.4]
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        pipelines(Config.model_validate(data))


def test_preprocess_reference_formula(config):
    image = np.random.default_rng(42).integers(0, 256, (73, 119, 3), dtype=np.uint8)
    reference = cv2.cvtColor(cv2.resize(image, (560, 560)), cv2.COLOR_BGR2RGB).astype(np.float32)
    mean = np.array([0.485 * 255, 0.456 * 255, 0.406 * 255], np.float32)
    std = np.array([0.229 * 255, 0.224 * 255, 0.225 * 255], np.float32)
    reference = ((reference - mean) / std).transpose(2, 0, 1)
    actual = pipelines(config).preprocess(image).tensors["input"]
    assert actual.dtype == np.float32 and actual.flags.c_contiguous
    np.testing.assert_allclose(actual[0], reference, atol=5e-7, rtol=1e-6)


def test_decode_reference_formula(config):
    pair = pipelines(config)
    prepared = pair.preprocess(np.zeros((373, 719, 3), np.uint8))
    rng = np.random.default_rng(9)
    boxes = rng.random((1, 300, 4), dtype=np.float32)
    logits = rng.standard_normal((1, 300, 2)).astype(np.float32)
    result = pair.postprocess(RawOutput({"dets": boxes, "labels": logits}, prepared))[0]
    probability = 1 / (1 + np.exp(-logits[0]))
    ids, scores = probability.argmax(-1), probability.max(-1)
    keep = scores >= 0.4
    b = boxes[0]
    reference = np.concatenate((b[:, :2] - b[:, 2:] / 2, b[:, :2] + b[:, 2:] / 2), axis=1) * [
        719,
        373,
        719,
        373,
    ]
    np.testing.assert_array_equal(result.class_ids, ids[keep])
    np.testing.assert_allclose(result.scores, scores[keep], atol=1e-7)
    np.testing.assert_array_equal(result.boxes, reference[keep].astype(int))


def test_letterbox_inverse_preserves_classes(config):
    data = config.model_dump()
    data["preprocess"]["steps"][0] = {
        "letterbox": {"width": 560, "height": 560, "rounding": "floor"}
    }
    options(data, "rfdetr_decode").update(integer_boxes=False)
    pair = pipelines(Config.model_validate(data))
    prepared = pair.preprocess(np.zeros((101, 200, 3), np.uint8))
    meta = prepared.metadata[0]
    expected = np.array([20, 10, 180, 90], np.float32)
    scaled = expected * [*meta.scale, *meta.scale] + [*meta.padding, *meta.padding]
    scaled /= [560, 560, 560, 560]
    box = np.concatenate(((scaled[:2] + scaled[2:]) / 2, scaled[2:] - scaled[:2]))
    boxes = np.tile(box.astype(np.float32), (1, 300, 1))
    logits = np.full((1, 300, 2), -1000, np.float32)
    logits[0, 0, 1] = 1000
    logits[0, 1, 0] = 1000
    result = pair.postprocess(RawOutput({"dets": boxes, "labels": logits}, prepared))[0]
    assert result.class_ids.tolist() == [1, 0]
    np.testing.assert_allclose(result.boxes[0], expected, atol=2e-5)


def test_no_detections_nonfinite_and_bad_normalization(config):
    pair = pipelines(config)
    prep = pair.preprocess(np.zeros((2, 2, 3), np.uint8))
    tensors = {
        "dets": np.zeros((1, 300, 4), np.float32),
        "labels": np.full((1, 300, 2), -100, np.float32),
    }
    result = pair.postprocess(RawOutput(tensors, prep))[0]
    assert result.boxes.shape == (0, 4) and result.scores.shape == (0,)
    tensors["labels"][0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        pair.postprocess(RawOutput(tensors, prep))
    data = config.model_dump()
    options(data, "normalize")["std"] = [1, 0, 1]
    with pytest.raises(ValueError):
        pipelines(Config.model_validate(data))
