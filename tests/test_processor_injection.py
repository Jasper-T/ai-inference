"""Complete processor injection preserves Pipeline compilation and request contracts."""

import numpy as np
import pytest

from ai_inference import BaseEngine, EngineRegistry, Model, Prediction, PreparedInput, Result


@pytest.mark.parametrize("results", [[], [object()], (object(),), None])
def test_custom_postprocessor_rejects_invalid_results(config, monkeypatch, results):
    monkeypatch.setattr(EngineRegistry, "create", lambda backend, config: BaseEngine(config))
    model = Model(config, postprocessor=lambda raw: results)
    prepared = model.preprocess(np.zeros((10, 10, 3), np.uint8))
    from ai_inference import RawOutput

    raw = RawOutput(
        {
            "dets": np.zeros((1, 300, 4), np.float32),
            "labels": np.zeros((1, 300, 2), np.float32),
        },
        prepared,
    )
    with pytest.raises(ValueError, match="one Result per real sample"):
        model.postprocess(raw)


def test_model_custom_pair_and_metadata(config, monkeypatch):
    calls = []

    class Engine(BaseEngine):
        def _infer(self, tensors):
            n = next(iter(tensors.values())).shape[0]
            calls.append(n)
            return {
                "dets": np.zeros((n, 300, 4), np.float32),
                "labels": np.zeros((n, 300, 2), np.float32),
            }

    monkeypatch.setattr(EngineRegistry, "create", lambda backend, config: Engine(config))

    def pre(images):
        return PreparedInput(
            {"input": np.zeros((len(images), 3, 560, 560), np.float32)},
            tuple({"index": i} for i in range(len(images))),
        )

    def post(raw) -> list[Result]:
        assert raw.prepared.metadata == tuple({"index": i} for i in range(5))
        assert raw.tensors["dets"].shape == (5, 300, 4)
        return [
            Prediction(
                np.empty((0, 4), np.float32), np.empty(0, np.float32), np.empty(0, np.int64), (1, 1)
            )
            for _ in raw.prepared.metadata
        ]

    model = Model(config, preprocessor=pre, postprocessor=post)
    assert model.preprocessor is pre and model.postprocessor is post
    assert len(model.predict([np.zeros((1, 1, 3), np.uint8)] * 5)) == 5
    assert calls == [4, 4]


def test_falsey_injected_processor_is_retained(config, monkeypatch):
    monkeypatch.setattr(EngineRegistry, "create", lambda backend, config: BaseEngine(config))

    class FalseyPreprocessor:
        def __bool__(self):
            return False

        def __call__(self, images):
            return PreparedInput({}, ())

    pre = FalseyPreprocessor()

    def post(raw):
        return []

    model = Model(config, preprocessor=pre, postprocessor=post)
    assert model.preprocessor is pre and model.postprocessor is post
