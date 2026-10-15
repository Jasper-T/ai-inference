"""No ONNX/GPU required: demonstrate batching using an explicit synthetic engine."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
import numpy as np
from model_contracts import load_synthetic_config

from ai_inference import BaseEngine, Config, EngineRegistry, Model, Prediction


@EngineRegistry.register("example_synthetic")
class SyntheticEngine(BaseEngine):
    def __init__(self, config):
        self.config = config
        self.calls = 0

    def _infer(self, tensors):
        self.calls += 1
        n = next(iter(tensors.values())).shape[0]
        boxes = np.tile(np.array([0.5, 0.5, 0.25, 0.25], np.float32), (n, 300, 1))
        logits = np.full((n, 300, 2), -20, np.float32)
        logits[:, 0, 0] = 2
        return {"dets": boxes, "labels": logits}


def main():
    config = load_synthetic_config(
        Path(__file__).resolve().parents[1] / "tests/fixtures/rfdetr_synthetic.yaml"
    )
    data = config.model_dump()
    data["runtime"]["backend"] = "example_synthetic"
    data["runtime"]["warmup"] = True
    data["runtime"]["warmup_runs"] = 3
    model = Model(Config.model_validate(data))
    images = [np.zeros((100, 200, 3), np.uint8) for _ in range(5)]
    results = model.predict(images)
    assert len(results) == 5 and model.engine.calls == 3 + 2
    assert isinstance(results[0], Prediction)
    assert results[0].class_ids.tolist() == [0]
    assert model.predict([]) == []
    print(
        "Synthetic smoke passed: 3 warmup calls; 5 images -> 2 batches of 4 -> 5 predictions; empty input supported."
    )


if __name__ == "__main__":
    main()
