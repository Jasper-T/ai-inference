import importlib
import importlib.util
import pkgutil
import subprocess
import sys
from pathlib import Path

import ai_inference


def test_stage_packages_and_public_exports_import():
    for name in ("contracts", "engines", "inference", "processing", "visualization"):
        importlib.import_module(f"ai_inference.{name}")
    for module in pkgutil.walk_packages(ai_inference.__path__, prefix="ai_inference."):
        if module.name.startswith("ai_inference.service.") and any(
            importlib.util.find_spec(name) is None for name in ("fastapi", "multipart")
        ):
            continue
        importlib.import_module(module.name)
    assert all(hasattr(ai_inference, name) for name in ai_inference.__all__)


def test_sdk_import_does_not_load_optional_http_dependencies():
    code = """
import importlib.abc
import sys

sys.path.insert(0, sys.argv[1])

class BlockHTTP(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'fastapi', 'starlette', 'uvicorn', 'multipart', 'httpx'}:
            raise ModuleNotFoundError(fullname)

sys.meta_path.insert(0, BlockHTTP())
import ai_inference
import ai_inference.service
assert callable(ai_inference.Model)
assert 'ai_inference.service.app' not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", code, str(Path(ai_inference.__file__).parent.parent)], check=True
    )


def test_retired_entry_points_are_removed():
    assert not hasattr(ai_inference, "BatchPredictor")
    for name in (
        "batching",
        "batching_validation",
        "predictor",
        "types",
        "validation",
        "registry",
        "context",
        "observability",
        "timing",
        "plotter",
        "pipeline",
    ):
        assert importlib.util.find_spec(f"ai_inference.{name}") is None
    batcher = importlib.import_module("ai_inference.inference.batcher")
    assert not hasattr(batcher, "execute")
    engine = importlib.import_module("ai_inference.engines.base")
    assert not hasattr(engine, "warmup_engine")
    assert not hasattr(engine, "_InferOnlyWarmupAdapter")
    assert not hasattr(ai_inference.Model, "_warmup")
