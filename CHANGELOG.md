# Changelog

Notable SDK and optional HTTP service changes are recorded here, newest first, following
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Breaking changes and migration instructions are included in the affected entries.

The SDK and optional HTTP service share the `ai-inference` distribution,
version and release notes. HTTP service changes are identified below.

## Unreleased

## 0.2.2 - 2026-10-15

### Added

- Explicit `onnx-gpu-cu12` (ORT >=1.21,<1.27) and `onnx-gpu-cu13` (ORT >=1.29,<1.32) extras; retain `onnx-gpu` as a compatibility entry point. Document supported Python/platform combinations, ARM64 CUDA 13 requirements and GPU installation checks.

### Fixed

- Gate GitHub Release and PyPI publication on the complete Linux and Windows acceptance workflow.
- Require Windows x64 Python 3.10 wheels during dependency locking and validate separate CPU/GPU installations on a native Windows runner; prevent selecting releases without cp310 Windows wheels.
- Require compatible Linux x86_64 / ARM64 and Windows x64 wheels for Python 3.10–3.12 when locking dependencies; exclude unsupported GPU platform/Python combinations and validate CPU installs on native Linux x86_64 / ARM64 runners. Run all package acceptance matrix entries even if another fails.
- Preserve real SSL context attributes in download test doubles so URL deadline regressions also run on Python 3.10.

## 0.2.1 - 2026-10-15

### Changed

- Move Ascend vendor wheel URLs to `requirements/ascend.txt`; the `ascend` extra remains empty for compatibility and requires separate runtime installation. Distribution metadata no longer includes direct URL dependencies.
- Declare the MIT license and project links, and add distribution metadata, clean installation and strict Twine checks before publication.
- Publish new version tags to PyPI through GitHub Actions Trusted Publishing, reusing validated GitHub Release artifacts; manual runs also support TestPyPI or GitHub-only publishing.

## 0.2.0 - 2026-10-15

### Added

- Optional single-model HTTP service under `ai_inference.service`; install `ai-inference[service,onnx-cpu]` for CPU serving. The `service` extra adds FastAPI, Uvicorn and multipart dependencies without affecting base SDK imports.
- File, Base64, URL and encoded-bytes prediction endpoints; detection/embedding responses, per-image SFace five-point inputs, health/model endpoints, request limits and single-request admission.
- HTTP boundary regression tests, including keypoint propagation through all four input forms; service usage, operations, review and release documentation.
- HTTP client example for file, Base64, URL and bytes requests, including optional SFace five-point metadata.
- Composable `Pipeline`, `StepRegistry`, and request-local `Context` / `Geometry`, with ordered YAML steps and validation of parameters, layout, color, device, and geometry.
- `Predictor` protocol, `DefaultPredictor`, and independent `Batcher` for splitting, padding, trimming, and merging fixed or dynamic batches in input order.
- Reusable image processing steps, model decoders, `l2_normalize`, and `filter_detections`; custom step registration and full preprocessor/postprocessor injection.
- Automatic ONNX Runtime and static OM tensor metadata discovery through `BaseEngine`; optional YAML input/output constraints.
- Initialization logs for original and resolved tensor shapes and validated batch settings; application-provided Loguru context.
- Independent `Plotter` for boxes, keypoints, masks, and feature summaries, with optional class names and application log context.
- Standalone benchmark, minimal model examples, multi-model PNG/JSON output, and CPU reference comparison/export tools.
- uv dependency management and lockfile, dev/test dependency groups, mutually exclusive `onnx-cpu`, `onnx-gpu`, and `ascend` extras, and a Python cache cleanup tool.
- GitHub Actions workflow for validating tagged SDK sources, building distribution artifacts, and publishing verified GitHub Releases.

### Changed

- Package HTTP service code with the SDK, use one version and CHANGELOG, and start it with `uvicorn ai_inference.service.app:app`. Replace the former standalone service installation with the `service` extra.
- Include service sources, documentation and regression tests in distribution/release checks; tagged releases validate the service extra alongside the SDK.
- Align the RF-DETR OM example with official Nano COCO defaults (384×384, batch=1, 91 class slots), retain the static AIPP contract, and remove legacy task-specific model names.
- Organized implementation into `contracts`, `engines`, `inference`, `processing`, and `visualization`. Update internal imports to the corresponding packages; prefer public exports.
- Require `preprocess.steps` / `postprocess.steps`. Keep only name/path in `model`, device settings in pipelines or steps, class counts in decoders, and display names in `Plotter`.
- Share CPU/GPU model configuration by changing providers; default YAML files discover model I/O through the engine.
- Use dynamic batch with a maximum of two samples for the RT-DETR example, symbolic candidate dimensions, and Pillow bilinear antialiased resizing. Deploy outputs require direct resizing without padding.
- Use 640×640 for YuNet 2023mar and separate SFace feature construction from L2 normalization.
- Support `cpu` / `cuda` / `trt` provider aliases and ordered lists. A single provider is strict; lists allow explicit initialization and operator fallback. Inference errors are never retried.
- Require actual OM tensor names instead of aliases. Custom engines must provide `config`, `infer`, and `warmup`.
- Include the SDK, optional HTTP service and supporting files in one distribution; include service tests in source distributions while excluding weights, build copies and bytecode.
- Document supported behavior, model sources, reproducible verification and the unified SDK/service release workflow.

### Fixed

- Enforce the URL download deadline on each socket read and send, TCP connection and TLS handshake; close response streams on all exit paths, including connection-close responses.
- Validate model support for face keypoints, degenerate alignment geometry and 112×112 aligned crops before inference; return indexed 422 errors without misclassifying genuine model failures.
- Reject static model batch sizes that conflict with YAML settings, reporting the tensor and corrective settings. Resolve symbolic batch dimensions according to the configured fixed or dynamic policy.
- Validate injected postprocessor result types and counts against the number of real input samples.
- Restore YuNet boxes and keypoints through composed resize/padding geometry.
- Validate RT-DETR output shapes at runtime and reject preprocessing incompatible with deploy coordinates.

### Removed

- `Adapter`, `AdapterRegistry`, legacy processors, internal forwarding modules, `BatchPredictor`, and `batching.execute`. Use `Model`, `Pipeline`, and `DefaultPredictor`.
- `Prediction.class_names`, SDK-generated request/instance IDs, timing state, and the `Model` request ID argument. Pass names to `Plotter`, bind application log fields, and use the standalone benchmark.
- The second `Model.postprocess` argument. Pass only `RawOutput`, which carries the original request metadata.
- RF-DETR per-class `thresholds`. Use a single `threshold` (default 0.4) and apply further class-specific filtering in the application; discarded candidates cannot be recovered.
- Model provenance, hash, and adoption records from SDK configuration and loading. Manage these in the deployment application.
- Legacy `cpu` / `ort` extras and the `test` installation extra. Use `onnx-cpu` / `onnx-gpu` / `ascend` for runtime dependencies and `uv sync --locked` for development. Ascend vendor wheels support Linux x86_64/aarch64 with CPython 3.10–3.12; install CANN and drivers separately.
- Legacy YAML/example entry points and the RF-DETR service comparison branch in `verify_predictions.py`; that tool now compares YOLO only. Use the current examples and RF-DETR export tool.

## 0.1.0 - 2026-10-13

### Added

- Initial Adapter / Processor / Engine architecture with shared `Model`, `Config`, registries, and `PreparedInput` / `RawOutput` boundaries.
- RF-DETR, YOLOv8 / YOLO11, RT-DETR deploy, YuNet 2023mar, and SFace 2021dec processing with ONNX Runtime and static OM backends, including RF-DETR AIPP inputs.
- `FaceInput`, `FeaturePrediction`, public result/input types, and optional detection keypoints and masks.
- Fixed/dynamic batches, `repeat_last` / `zeros` padding, ordered results, and configurable warmup.
- Model provenance and optional ONNX hash validation.
- Model configuration, examples, and synthetic regression checks.

### Changed

- Standardized the distribution name as `ai-inference` and the import name as `ai_inference`; update imports and reinstall when migrating from earlier names.

### Fixed

- Validate requested and loaded ONNX Runtime providers and model I/O names, dtypes, ranks, and static dimensions.
- Disable automatic runtime retry and prohibit CPU graph fallback unless CPU is explicitly configured.
