import numpy as np
import pytest
from model_contracts import load_synthetic_config
from pipeline_helpers import options, pipelines

from ai_inference import Config
from ai_inference.processing.common import image_list, letterbox
from ai_inference.processing.geometry import clip_boxes, cxcywh_to_xyxy, nms, restore_boxes


def test_image_contract_and_empty_batch():
    image = np.zeros((3, 5, 3), np.uint8)
    assert image_list(image)[0] is image
    assert image_list([]) == []
    for invalid in [
        np.zeros((0, 5, 3), np.uint8),
        np.zeros((3, 5), np.uint8),
        image.astype(float),
        None,
    ]:
        with pytest.raises(ValueError):
            image_list([invalid])
    with pytest.raises(TypeError):
        image_list((image,))  # pyright: ignore[reportArgumentType] - intentional invalid input


@pytest.mark.parametrize("rounding", ["floor", "ultralytics"])
def test_odd_letterbox_and_inverse(rounding):
    image = np.full((3, 7, 3), 255, np.uint8)
    canvas, t = letterbox(image, 10, 10, rounding=rounding)
    assert t.padding == (0, 3) and t.scale == (10 / 7, 4 / 3) and t.gain == 10 / 7
    assert (canvas[:3] == 114).all() and (canvas[3:7] == 255).all() and (canvas[7:] == 114).all()
    original = np.array([[0, 0, 7, 3]], np.float64)
    for scale in [t.scale, t.gain]:
        multiplier = [*scale, *scale] if isinstance(scale, tuple) else scale
        encoded = original * multiplier + [*t.padding, *t.padding]
        np.testing.assert_allclose(restore_boxes(encoded, scale, t.padding), original)
    small, st = letterbox(image, 10, 10, scale_up=False, rounding=rounding)
    assert st.gain == 1 and st.padding == (1, 3)
    assert small.shape == (10, 10, 3)


def test_boxes_and_nms():
    raw = np.array([[5, 5, -2, 4]], np.float32)
    np.testing.assert_array_equal(cxcywh_to_xyxy(raw, clamp_wh=True), [[5, 3, 5, 7]])
    np.testing.assert_array_equal(
        clip_boxes(np.array([[-1, -2, 30, 40]]), (10, 20)), [[0, 0, 20, 10]]
    )
    boxes = np.array([[0, 0, 10, 10]] * 3, np.float64)
    scores = np.array([0.9, 0.9, 0.8])
    ids = np.array([0, 0, 1])
    np.testing.assert_array_equal(nms(boxes, scores, ids, 0.7, 300, False), [0, 2])
    np.testing.assert_array_equal(nms(boxes, scores, ids, 0.7, 300, True), [0])
    assert nms(np.empty((0, 4)), np.empty(0), np.empty(0), 0.7, 300, False).shape == (0,)


def test_normalization_schema_rejected_before_execution(config):
    data = config.model_dump()
    options(data, "normalize")["std"] = [1, 0, 1]
    with pytest.raises(ValueError):
        pipelines(Config.model_validate(data))


def test_tensor_dtype_must_match_pipeline():
    from pathlib import Path

    cfg = load_synthetic_config(Path(__file__).resolve().parents[1] / "configs/yolo11.yaml")
    data = cfg.model_dump()
    data["inputs"][0]["dtype"] = "uint8"
    with pytest.raises(ValueError, match="dtype"):
        pipelines(Config.model_validate(data))
