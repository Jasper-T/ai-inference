"""Model-specific output algorithms reused as explicit pipeline operations."""

from dataclasses import replace

from ai_inference.processing.decoders import rfdetr, rtdetr, sface, yolo, yunet
from ai_inference.processing.pipeline import Step, StepRegistry


def image_dimensions(config, *, nhwc=False):
    spec = config.inputs[0]
    if len(spec.shape) != 4:
        raise ValueError("model image input must have rank 4")
    dimensions = spec.shape[1:3] if nhwc else spec.shape[2:]
    if any(d is None for d in dimensions):
        raise ValueError("built-in model decoding requires concrete image dimensions")
    return dimensions


def checked_outputs(config, expected, *, allow_dynamic=False):
    actual = {s.name: s for s in config.outputs}
    if set(actual) != set(expected):
        raise ValueError("model output tensor names differ from decoder contract")
    for name, (dtype, shape) in expected.items():
        spec = actual[name]
        actual_shape = spec.shape[1:]
        shape_matches = actual_shape == shape
        if allow_dynamic:
            shape_matches = len(actual_shape) == len(shape) and all(
                got is None or want is None or got == want for got, want in zip(actual_shape, shape)
            )
        if spec.dtype != dtype or not shape_matches:
            raise ValueError(f"model output dtype/shape mismatch: {name}")


def decode_step(processor, *, geometry=True, uniform=False, color=None):
    def run(context):
        if geometry:
            for meta in context.data.prepared.metadata:
                if hasattr(meta, "restorable") and not meta.restorable:
                    raise ValueError("detection decoder cannot restore face-aligned coordinates")
                if uniform and getattr(meta, "gain", None) is None:
                    raise ValueError("YOLO restoration requires uniform resize/letterbox geometry")
        context.data = processor(context.data)
        context.kind = "results"

    return Step(
        "raw",
        "results",
        run,
        lambda state: None,
        required_color=color,
        geometry_mode=("uniform" if uniform else "actual") if geometry else None,
    )


@StepRegistry.register("yolo_decode")
def yolo_decode(config, **options):
    p = yolo.Options.model_validate(options).resolve_output(config)
    if len(config.inputs) != 1 or len(config.outputs) != 1:
        raise ValueError("YOLO detection requires one image input and one raw output")
    inp, out = config.inputs[0], config.outputs[0]
    if len(inp.shape) != 4 or inp.shape[1] != 3 or inp.dtype not in ("float32", "float16"):
        raise ValueError("YOLO input must be float32/float16 NCHW")
    axis = 1 if p.layout == "NCQ" else 2
    if (
        out.dtype not in ("float32", "float16")
        or len(out.shape) != 3
        or out.shape[axis] != 4 + p.num_classes
    ):
        raise ValueError("YOLO output must be configured [N,4+C,Q] or [N,Q,4+C]")
    return decode_step(yolo.Decoder(config, p), uniform=True, color="RGB")


@StepRegistry.register("rfdetr_decode")
def rfdetr_decode(config, **options):
    p = rfdetr.Options.model_validate(options)
    if len(config.inputs) != 1 or p.boxes_name == p.logits_name:
        raise ValueError("RF-DETR requires one input and distinct output names")
    inp = config.inputs[0]
    aipp = inp.dtype == "uint8"
    h, w = image_dimensions(config, nhwc=aipp)
    if inp.dtype not in ("uint8", "float32") or inp.shape[1:] != ((h, w, 3) if aipp else (3, h, w)):
        raise ValueError("RF-DETR input must be float32 NCHW or AIPP uint8 NHWC")
    boxes = next((s for s in config.outputs if s.name == p.boxes_name), None)
    if boxes is None or len(boxes.shape) != 3:
        raise ValueError("RF-DETR boxes must be [N,Q,4]")
    q = boxes.shape[1]
    checked_outputs(
        config, {p.boxes_name: ("float32", (q, 4)), p.logits_name: ("float32", (q, p.num_classes))}
    )
    return decode_step(rfdetr.Decoder(config, p, width=w, height=h), color="BGR" if aipp else "RGB")


@StepRegistry.register("rtdetr_decode")
def rtdetr_decode(config, **options):
    p = rtdetr.Options.model_validate(options)
    if len(config.inputs) != 2:
        raise ValueError("RT-DETR requires image and original target sizes inputs")
    images = [s for s in config.inputs if len(s.shape) == 4]
    sizes = [s for s in config.inputs if len(s.shape) == 2]
    if (
        len(images) != 1
        or images[0].shape[1] != 3
        or images[0].dtype != "float32"
        or len(sizes) != 1
        or sizes[0].shape[1:] != (2,)
        or sizes[0].dtype not in ("float32", "int64")
    ):
        raise ValueError("invalid RT-DETR image/target size contract")
    labels = next((s for s in config.outputs if s.name == p.labels_name), None)
    if (
        labels is None
        or len(labels.shape) != 2
        or len({p.labels_name, p.boxes_name, p.scores_name}) != 3
    ):
        raise ValueError("RT-DETR requires distinct labels/boxes/scores outputs")
    # Exporters may leave query counts and the final box axis symbolic.
    # Known dimensions must agree; the decoder checks actual shapes at runtime.
    counts = {s.shape[1] for s in config.outputs if len(s.shape) > 1 and s.shape[1] is not None}
    if len(counts) > 1:
        raise ValueError("RT-DETR output candidate counts must agree")
    k = next(iter(counts), None)
    checked_outputs(
        config,
        {
            p.labels_name: ("int64", (k,)),
            p.boxes_name: ("float32", (k, 4)),
            p.scores_name: ("float32", (k,)),
        },
        allow_dynamic=True,
    )
    return replace(decode_step(rtdetr.Decoder(config, p), color="RGB"), geometry_mode="resize")


@StepRegistry.register("yunet_decode")
def yunet_decode(config, **options):
    p = yunet.Options.model_validate(options)
    h, w = image_dimensions(config)
    if (
        len(config.inputs) != 1
        or config.inputs[0].dtype != "float32"
        or config.inputs[0].shape[1] != 3
        or h % 32
        or w % 32
    ):
        raise ValueError("YuNet requires float32 NCHW, dimensions divisible by 32")
    checked_outputs(
        config,
        {
            f"{kind}_{stride}": ("float32", ((h // stride) * (w // stride), channels))
            for kind, channels in yunet.CHANNELS.items()
            for stride in yunet.STRIDES
        },
    )
    return decode_step(yunet.Decoder(config, p, width=w, height=h), color="BGR")


@StepRegistry.register("sface_features")
def sface_features(config, **options):
    p = sface.Options.model_validate(options)
    if (
        len(config.inputs) != 1
        or config.inputs[0].dtype != "float32"
        or config.inputs[0].shape[1:] != (3, 112, 112)
    ):
        raise ValueError("SFace requires float32 NCHW 112x112 input")
    checked_outputs(config, {p.output_name: ("float32", (p.dimensions,))})
    return decode_step(sface.Decoder(config, p), geometry=False, color="RGB")
