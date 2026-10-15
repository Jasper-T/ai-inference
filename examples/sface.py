"""Run local YuNet/SFace weights on tests/bus.jpg when a face is detected."""

from pathlib import Path

import cv2

from ai_inference import FaceInput, Model, Plotter, Prediction

ROOT = Path(__file__).resolve().parents[1]
yaml_path = ROOT / "configs/sface.yaml"  # YAML points to this repository's models/.
model = Model(yaml_path)
image_path = ROOT / "tests/bus.jpg"
image = cv2.imread(str(image_path))
if image is None:
    raise FileNotFoundError(image_path)
faces = Model(ROOT / "configs/yunet.yaml").predict(image)[0]
if not isinstance(faces, Prediction) or faces.keypoints is None:
    raise ValueError("YuNet must return a detection result with face keypoints")
result = model.predict(FaceInput(image, faces.keypoints[0]))[0] if len(faces.boxes) else None
Plotter().save(
    ROOT / "results/example_sface.png",
    image,
    result,
    context=faces,
    names={0: "face"},
    note="SFace not run: no face detected" if result is None else "",
)
