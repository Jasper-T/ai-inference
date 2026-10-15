"""Fixed synthetic tensor contracts for tests and smoke, independent of model files.

Production YAML uses engine discovery. These contracts let pipeline and fake-engine
checks run without loading pretrained models or vendor runtimes.
"""

from ai_inference import Config

CONTRACTS = {
    "rtdetr": {
        "inputs": [("images", "float32", (1, 3, 640, 640)), ("orig_target_sizes", "int64", (1, 2))],
        "outputs": [
            ("labels", "int64", (1, 300)),
            ("boxes", "float32", (1, 300, 4)),
            ("scores", "float32", (1, 300)),
        ],
    },
    "sface": {
        "inputs": [("data", "float32", (1, 3, 112, 112))],
        "outputs": [("fc1", "float32", (1, 128))],
    },
    "yolo11": {
        "inputs": [("images", "float32", (1, 3, 640, 640))],
        "outputs": [("output0", "float32", (1, 84, 8400))],
    },
    "yunet": {
        "inputs": [("input", "float32", (1, 3, 640, 640))],
        "outputs": [
            ("cls_8", "float32", (1, 6400, 1)),
            ("cls_16", "float32", (1, 1600, 1)),
            ("cls_32", "float32", (1, 400, 1)),
            ("obj_8", "float32", (1, 6400, 1)),
            ("obj_16", "float32", (1, 1600, 1)),
            ("obj_32", "float32", (1, 400, 1)),
            ("bbox_8", "float32", (1, 6400, 4)),
            ("bbox_16", "float32", (1, 1600, 4)),
            ("bbox_32", "float32", (1, 400, 4)),
            ("kps_8", "float32", (1, 6400, 10)),
            ("kps_16", "float32", (1, 1600, 10)),
            ("kps_32", "float32", (1, 400, 10)),
        ],
    },
    "rfdetr_om_synthetic": {
        "inputs": [("input", "uint8", (4, 560, 560, 3))],
        "outputs": [("dets", "float32", (4, 300, 4)), ("labels", "float32", (4, 300, 2))],
    },
    "rfdetr_synthetic": {
        "inputs": [("input", "float32", (4, 3, 560, 560))],
        "outputs": [("dets", "float32", (4, 300, 4)), ("labels", "float32", (4, 300, 2))],
    },
}


def load_synthetic_config(path):
    config = Config.from_yaml(path)
    data = config.model_dump()
    # Synthetic graphs use fixed tensors; dynamic RT-DETR is tested separately.
    if path.stem == "rtdetr":
        data["runtime"]["batch"].update(mode="fixed", size=1)
    for kind, specs in CONTRACTS[path.stem].items():
        data[kind] = [dict(name=name, dtype=dtype, shape=shape) for name, dtype, shape in specs]
    return Config.model_validate(data)
