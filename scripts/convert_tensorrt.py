#!/usr/bin/env python3
"""Build a serialized TensorRT 10 engine from ONNX on the target NVIDIA GPU."""

import argparse
from pathlib import Path


def parse_profile(value):
    """Parse name=min:opt:max;other=min:opt:max with x-separated dimensions."""
    result = {}
    try:
        for entry in value.split(";"):
            name, dimensions = entry.split("=", 1)
            shapes = tuple(
                tuple(int(dim) for dim in shape.split("x")) for shape in dimensions.split(":")
            )
            if (
                not name
                or name in result
                or len(shapes) != 3
                or len({len(shape) for shape in shapes}) != 1
                or any(dim <= 0 for shape in shapes for dim in shape)
                or any(not low <= opt <= high for low, opt, high in zip(*shapes))
            ):
                raise ValueError
            result[name] = shapes
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "profile must be 'name=min:opt:max;other=min:opt:max', e.g. 'images=2x3x640x640:4x3x640x640:8x3x640x640'"
        ) from exc
    return result


def configure_profiles(builder, network, config, profiles):
    inputs = {network.get_input(i).name: network.get_input(i) for i in range(network.num_inputs)}
    if any(tensor.is_shape_tensor for tensor in inputs.values()):
        raise ValueError(
            "shape tensor inputs are outside this converter's supported batch contract"
        )
    dynamic = {name for name, tensor in inputs.items() if -1 in tensor.shape}
    if dynamic and not profiles:
        raise ValueError("dynamic ONNX inputs require at least one --profile")
    for shapes in profiles:
        if shapes.keys() - inputs.keys() or dynamic - shapes.keys():
            raise ValueError(
                "each profile must cover every dynamic input and contain no unknown inputs"
            )
        profile = builder.create_optimization_profile()
        for name, tensor in inputs.items():
            declared = tuple(tensor.shape)
            bounds = shapes.get(name, (declared,) * 3)
            if any(
                len(shape) != len(declared)
                or any(actual != fixed for actual, fixed in zip(shape, declared) if fixed != -1)
                for shape in bounds
            ):
                raise ValueError(f"profile conflicts with ONNX dimensions: {name}")
            profile.set_shape(name, *bounds)
        if config.add_optimization_profile(profile) < 0:
            raise RuntimeError("TensorRT rejected optimization profile")


def build(args, trt):
    logger = trt.Logger(trt.Logger.WARNING)
    if not trt.init_libnvinfer_plugins(logger, ""):
        raise RuntimeError("TensorRT plugin initialization failed")
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(args.onnx.resolve())):
        errors = "\n".join(str(parser.get_error(index)) for index in range(parser.num_errors))
        raise RuntimeError(f"ONNX parsing failed:\n{errors}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace_mib * 1024**2)
    config.clear_flag(trt.BuilderFlag.TF32)
    if args.precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    configure_profiles(builder, network, config, args.profile)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed")
    return bytes(serialized)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("onnx", type=Path)
    parser.add_argument("engine", type=Path)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--workspace-mib", type=int, default=1024)
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--profile", type=parse_profile, action="append", default=[])
    args = parser.parse_args(argv)
    if args.device_id < 0 or args.workspace_mib <= 0:
        parser.error("device-id must be nonnegative and workspace-mib must be positive")
    if args.onnx.resolve() == args.engine.resolve():
        parser.error("ONNX input and engine output must be different files")
    import tensorrt as trt
    from cuda.bindings import driver

    if trt.__version__.split(".")[0] != "10":
        raise RuntimeError("This converter requires TensorRT 10.x")

    def call(name, *values):
        status, *results = getattr(driver, name)(*values)
        if int(status):
            raise RuntimeError(f"CUDA {name} failed: {status}")
        return results[0] if results else None

    call("cuInit", 0)
    device = call("cuDeviceGet", args.device_id)
    context = call("cuDevicePrimaryCtxRetain", device)
    try:
        call("cuCtxPushCurrent", context)
        try:
            serialized = build(args, trt)
        finally:
            call("cuCtxPopCurrent")
    finally:
        call("cuDevicePrimaryCtxRelease", device)
    # Build failures never overwrite an existing engine.
    args.engine.write_bytes(serialized)
    print(f"Wrote {args.engine} ({len(serialized)} bytes)")


if __name__ == "__main__":
    main()
