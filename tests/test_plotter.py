import cv2
import numpy as np
import pytest

from ai_inference import FeaturePrediction, Plotter, Prediction


def detection(shape=(120, 160)):
    return Prediction(
        np.array([[40, 40, 100, 90]], np.float32),
        np.array([0.8], np.float32),
        np.array([0], np.int64),
        shape,
        keypoints=np.array([[[60, 60], [80, 60]]], np.float32),
    )


def test_independent_render_preserves_input_and_prediction(monkeypatch, tmp_path):
    from ai_inference import Model

    monkeypatch.setattr(
        Model, "predict", lambda *a, **k: pytest.fail("plotter must not call predictor")
    )
    image = np.zeros((120, 160, 3), np.uint8)
    result = detection()
    boxes = result.boxes.copy()
    rendered = Plotter().render(image, result)
    assert rendered.shape == image.shape and rendered.dtype == np.uint8
    assert rendered[60, 60].any() and rendered[90 - 1, 40].any()
    assert not image.any()
    np.testing.assert_array_equal(result.boxes, boxes)
    path = Plotter().save(tmp_path / "nested/out.png", image, result)
    np.testing.assert_array_equal(cv2.imread(str(path)), rendered)


def test_feature_context_and_empty_status():
    image = np.zeros((120, 160, 3), np.uint8)
    feature = FeaturePrediction(np.array([0.6, 0.8], np.float32), (120, 160))
    rendered = Plotter().render(image, feature, context=detection(), title="SFace")
    assert rendered[60, 60].any()
    status = Plotter().render(image, None, title="SFace", note="Not run: no valid face landmarks")
    assert status.any() and not image.any()


def test_masks_clipping_and_validation():
    image = np.zeros((120, 160, 3), np.uint8)
    result = Prediction(
        np.array([[-10, 40, 200, 150]], np.float32),
        np.array([1], np.float32),
        np.array([0]),
        (120, 160),
        masks=np.ones((1, 120, 160), bool),
    )
    rendered = Plotter().render(image, result)
    assert rendered[-2, 80].any()
    assert not image.any()
    with pytest.raises(ValueError):
        Plotter().render(image, detection((1, 2)))
    with pytest.raises(ValueError):
        Plotter(mask_alpha=2)
    with pytest.raises(ValueError):
        Plotter().render(image.astype(np.float32))


@pytest.mark.parametrize(
    "names,ident,expected",
    [
        (None, 7, "7"),
        (["person", "face"], 1, "face"),
        (["person"], 7, "7"),
        ({0: "person"}, 0, "person"),
        ({0: "person"}, 7, "7"),
    ],
)
def test_optional_names(names, ident, expected):
    assert Plotter.label(ident, names) == expected


def test_names_change_only_rendered_text(monkeypatch):
    image = np.zeros((120, 160, 3), np.uint8)
    prediction = detection()
    labels = []
    original = cv2.putText

    def put_text(canvas, text, *args, **kwargs):
        labels.append(text)
        return original(canvas, text, *args, **kwargs)

    monkeypatch.setattr(cv2, "putText", put_text)
    Plotter().render(image, prediction)
    assert "0 0.8000" in labels
    labels.clear()
    Plotter().render(image, prediction, names={0: "person"})
    assert "person 0.8000" in labels
    np.testing.assert_array_equal(prediction.class_ids, [0])
    assert not hasattr(prediction, "class_names") and not image.any()


def test_class_colors_do_not_wrap_landmark_palette():
    image = np.zeros((120, 160, 3), np.uint8)
    result = Prediction(
        np.array([[20, 50, 60, 100], [90, 50, 130, 100]], np.float32),
        np.array([0.9, 0.8], np.float32),
        np.array([0, 5], np.int64),
        (120, 160),
    )
    rendered = Plotter().render(image, result)
    assert not np.array_equal(rendered[70, 20], rendered[70, 90])
    assert Plotter.color(0) == (255, 42, 4)
    assert Plotter.color(5) == (221, 111, 255)
    assert len({Plotter.color(i) for i in range(20)}) == 20
    assert Plotter.color(20) == Plotter.color(0)
    # A class has the same color whether it appears first or second in the image.
    reversed_result = Prediction(
        result.boxes[::-1], result.scores[::-1], result.class_ids[::-1], result.original_shape
    )
    np.testing.assert_array_equal(Plotter().render(image, reversed_result)[60:], rendered[60:])


def test_no_header_by_default_for_detections_or_features():
    image = np.full((120, 160, 3), 100, np.uint8)
    for result in (detection(), FeaturePrediction(np.array([0.6, 0.8], np.float32), (120, 160))):
        rendered = Plotter().render(image, result)
        np.testing.assert_array_equal(rendered[:20], image[:20])
        assert not np.array_equal(
            Plotter().render(image, result, show_summary=True)[:20], image[:20]
        )
