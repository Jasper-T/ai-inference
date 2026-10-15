#!/usr/bin/env python3
"""One image, each ONNX family, independent Plotter, local results directory."""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=ROOT / "tests/bus.jpg")
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--rfdetr-model", type=Path, default=ROOT / "models/rfdetr-nano.onnx")
    args = parser.parse_args()
    import cv2
    import numpy as np
    import onnxruntime as ort

    ort.set_default_logger_severity(3)
    from verify_predictions import digest

    from ai_inference import Config, FaceInput, Model, Plotter, Prediction

    image = cv2.imread(str(args.image))
    if image is None:
        parser.error(f"cannot decode image: {args.image}")
    args.results.mkdir(parents=True, exist_ok=True)
    plotter = Plotter()
    records = []
    faces = None
    entries = [
        ("yolo11", "yolo11.yaml", ROOT / "models/yolo11n.onnx"),
        ("rfdetr", "rfdetr.yaml", args.rfdetr_model),
        ("rtdetr", "rtdetr.yaml", ROOT / "models/rtdetr-deployed-v1.onnx"),
        ("yunet", "yunet.yaml", ROOT / "models/face_detection_yunet_2023mar.onnx"),
        ("sface", "sface.yaml", ROOT / "models/face_recognition_sface_2021dec.onnx"),
    ]
    for name, config_name, path in entries:
        record: dict[str, Any] = dict(
            model=name,
            image=str(args.image.resolve()),
            image_shape=list(image.shape[:2]),
            image_sha256=digest(args.image),
            forward_calls=0,
        )
        annotation = args.results / f"{name}.png"
        try:
            if not path.is_file():
                raise FileNotFoundError(f"local model missing: {path}")
            record.update(model_path=str(path.resolve()), model_sha256=digest(path))
            data = Config.from_yaml(ROOT / "configs" / config_name).model_dump()
            data["model"].update(path=path.resolve())
            data["runtime"].update(
                warmup=False, options={"providers": "cpu", "intra_op_num_threads": 1}
            )
            model = Model(Config.model_validate(data))
            source = (
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                if data["preprocess"].get("source_color", "BGR") == "RGB"
                else image
            )
            if name == "sface" and (faces is None or not len(faces.boxes)):
                record.update(status="not_run", reason="no_valid_face_landmarks", embedding=None)
                plotter.save(annotation, image)
            else:
                if name == "sface":
                    if faces is None or faces.keypoints is None:
                        raise ValueError("SFace requires detected face landmarks")
                    index = int(np.argmax(faces.scores))
                    inputs = FaceInput(source, faces.keypoints[index])
                else:
                    inputs = source
                started = time.perf_counter()
                prepared = model.preprocess(inputs)
                record["forward_calls"] = 1
                raw = model.infer(prepared)
                result = model.postprocess(raw)[0]
                record.update(
                    status="ok",
                    seconds=time.perf_counter() - started,
                    providers=list(model.engine.providers),
                    physical_batch_size=(
                        data["runtime"]["batch"]["size"]
                        if data["runtime"]["batch"]["mode"] == "fixed"
                        else prepared.count
                    ),
                )
                if isinstance(result, Prediction):
                    record.update(
                        boxes=result.boxes.tolist(),
                        scores=result.scores.tolist(),
                        class_ids=result.class_ids.tolist(),
                        keypoints=None if result.keypoints is None else result.keypoints.tolist(),
                    )
                    plotter.save(annotation, image, result)
                    if name == "yunet":
                        faces = result
                else:
                    if name != "sface" or faces is None or faces.keypoints is None:
                        raise ValueError("feature annotation requires detected face landmarks")
                    index = int(np.argmax(faces.scores))
                    context = Prediction(
                        faces.boxes[index : index + 1],
                        faces.scores[index : index + 1],
                        faces.class_ids[index : index + 1],
                        faces.original_shape,
                        keypoints=faces.keypoints[index : index + 1],
                    )
                    record.update(
                        face_index=index,
                        embedding=result.embedding.tolist(),
                        dimensions=len(result.embedding),
                        face_box=context.boxes[0].tolist(),
                        keypoints=faces.keypoints[index].tolist(),
                    )
                    plotter.save(annotation, image, result, context=context)
            print(
                f"{name}: {record['status']}, forward calls={record['forward_calls']}", flush=True
            )
        except Exception as exc:
            record.update(status="error", error=f"{type(exc).__name__}: {exc}")
            plotter.save(annotation, image)
            print(f"{name}: ERROR {exc}", flush=True)
        record["annotation"] = annotation.name
        (args.results / f"{name}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2))
        records.append(record)
    (args.results / "summary.json").write_text(json.dumps(records, ensure_ascii=False, indent=2))
    print(f"Results: {args.results.resolve()}")
    return 1 if any(r["status"] == "error" for r in records) else 0


if __name__ == "__main__":
    raise SystemExit(main())
