"""Read canonical processing settings for manual reference comparisons.

This is inspection only: actual requests always execute Model's configured path.
Custom pipelines need their own reference contract instead of guessed settings.
"""

from types import SimpleNamespace

from ai_inference.processing import step_options
from ai_inference.processing.decoders import rtdetr, sface, yolo, yunet


def reference_settings(config, family):
    modules = {"yolo": yolo, "rtdetr": rtdetr, "yunet": yunet, "sface": sface}
    module = modules[family]
    expected = {
        "yolo": ["resize", "padding", "color", "normalize", "layout", "pack"],
        "rtdetr": ["color", "resize", "normalize", "layout", "pack"],
        "yunet": ["resize", "color", "cast", "layout", "pack"],
        "sface": ["face_align", "color", "cast", "layout", "pack"],
    }
    names = [next(iter(entry)) for entry in config.preprocess["steps"]]
    if names != expected[family]:
        raise ValueError(f"reference comparison requires canonical {family} preprocessing steps")
    decode = {
        "yolo": "yolo_decode",
        "rtdetr": "rtdetr_decode",
        "yunet": "yunet_decode",
        "sface": "sface_features",
    }[family]
    post_names = [next(iter(entry)) for entry in config.postprocess["steps"]]
    if post_names not in ([decode], [decode, "l2_normalize"] if family == "sface" else [decode]):
        raise ValueError("reference comparison requires canonical postprocessing steps")
    pack = step_options(config, "pack")
    pre = dict(
        input_name=pack["input_name"], source_color=config.preprocess.get("source_color", "BGR")
    )
    if family == "sface":
        pre.update(step_options(config, "face_align"))
    else:
        resize = step_options(config, "resize")
        pre.update(width=resize["width"], height=resize["height"])
        if family == "yolo":
            if (
                not resize.get("keep_ratio")
                or step_options(config, "padding").get("alignment", "center") != "center"
            ):
                raise ValueError("YOLO reference requires proportional resize and centered padding")
            pre.update(
                scale_up=resize.get("scale_up", True),
                pad_value=step_options(config, "padding").get("value", 114),
            )
        elif resize.get("keep_ratio", False):
            raise ValueError("reference requires stretching resize")
        if family == "rtdetr":
            pre.update(
                resize_backend=resize.get("backend", "opencv"), sizes_name=pack["sizes_name"]
            )
        if family in ("yolo", "rtdetr"):
            norm = step_options(config, "normalize")
            if norm.get("divisor") != 255 or norm.get("scale") is not None:
                raise ValueError("reference requires divisor=255 normalization")
    post = module.Options.model_validate(step_options(config, decode)).model_dump()
    return SimpleNamespace(**pre), SimpleNamespace(**post)
