"""Run models/rfdetr-nano.onnx on tests/bus.jpg."""

from pathlib import Path

import cv2

from ai_inference import Model, Plotter

ROOT = Path(__file__).resolve().parents[1]
yaml_path = ROOT / "configs/rfdetr.yaml"  # YAML points to this repository's models/.
model = Model(yaml_path)
image_path = ROOT / "tests/bus.jpg"
image = cv2.imread(str(image_path))
if image is None:
    raise FileNotFoundError(image_path)
result = model.predict(image)[0]
Plotter().save(ROOT / "results/example_rfdetr.png", image, result, names={1: "person", 6: "bus"})
