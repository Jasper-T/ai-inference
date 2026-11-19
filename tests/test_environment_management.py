"""Check exported backend selections and reproducible vendor wheel requirements."""

import importlib.util
import os
import shutil
import subprocess
from itertools import combinations
from pathlib import Path
from urllib.parse import urldefrag

import pytest
from packaging.requirements import Requirement
from packaging.version import Version

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]


def test_vendor_requirements_generated_from_extra_sources_and_lock():
    spec = importlib.util.spec_from_file_location(
        "ascend_requirements", ROOT / "scripts/ascend_requirements.py"
    )
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    assert (ROOT / "requirements/ascend.txt").read_text() == generator.render(project, lock)


@pytest.mark.parametrize("python", ["3.10", "3.11", "3.12"])
@pytest.mark.parametrize("arch", ["x86_64", "aarch64"])
def test_selected_vendor_builds_match_locked_hashes(python, arch):
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    environment = {
        "sys_platform": "linux",
        "platform_machine": arch,
        "python_version": python,
        "python_full_version": python + ".1",
        "implementation_name": "cpython",
    }
    selected = [
        r
        for raw in project["project"]["optional-dependencies"]["ascend"]
        if (r := Requirement(raw)).marker.evaluate(environment)
    ]
    assert {r.name for r in selected} == {"aclruntime", "ais-bench"}
    pip_requirements = [
        Requirement(line)
        for line in (ROOT / "requirements/ascend.txt").read_text().splitlines()
        if line and not line.startswith("#")
    ]
    for requirement in selected:
        assert requirement.url is None
        sources = [
            source
            for source in project["tool"]["uv"]["sources"][requirement.name]
            if source["extra"] == "ascend"
            and Requirement("vendor; " + source["marker"]).marker.evaluate(environment)
        ]
        assert len(sources) == 1
        selected_pip = [
            r
            for r in pip_requirements
            if r.name == requirement.name and r.marker.evaluate(environment)
        ]
        assert len(selected_pip) == 1
        url, fragment = urldefrag(selected_pip[0].url)
        assert url == sources[0]["url"]
        algorithm, digest = fragment.split("=", 1)
        assert algorithm == "sha256" and len(digest) == 64
        int(digest, 16)
        package = next(
            p
            for p in lock["package"]
            if p["name"] == requirement.name and p["source"] == {"url": url}
        )
        assert package["version"] == "0.0.2"
        assert any(w["url"] == url and w["hash"] == "sha256:" + digest for w in package["wheels"])


@pytest.mark.skipif(shutil.which("uv") is None, reason="backend export checks require uv")
@pytest.mark.parametrize(
    "arguments,required,forbidden",
    [
        (
            ["--only-group", "dev"],
            {"pytest", "ruff", "twine"},
            {"onnxruntime", "onnxruntime-gpu", "aclruntime", "ais-bench"},
        ),
        (
            ["--extra", "onnx-cpu"],
            {"pytest", "onnxruntime"},
            {"onnxruntime-gpu", "aclruntime", "ais-bench"},
        ),
        *[
            (
                ["--extra", extra],
                {"pytest", "onnxruntime-gpu"},
                {"onnxruntime", "aclruntime", "ais-bench"},
            )
            for extra in ("onnx-gpu-cu11", "onnx-gpu-cu12", "onnx-gpu-cu13")
        ],
        (
            ["--extra", "ascend"],
            {"pytest", "aclruntime", "ais-bench"},
            {"onnxruntime", "onnxruntime-gpu"},
        ),
        (
            ["--extra", "tensorrt-cu12"],
            {
                "pytest",
                "tensorrt-cu12",
                "tensorrt-cu12-bindings",
                "tensorrt-cu12-libs",
                "cuda-bindings",
            },
            {"onnxruntime", "onnxruntime-gpu", "aclruntime", "ais-bench"},
        ),
    ],
)
def test_backend_exports_keep_tools_without_other_runtimes(
    tmp_path, arguments, required, forbidden
):
    env = dict(os.environ, UV_CACHE_DIR=str(tmp_path / "uv-cache"), UV_OFFLINE="1")
    result = subprocess.run(
        ["uv", "export", "--frozen", "--no-hashes", "--no-emit-project", *arguments],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    requirements = [
        Requirement(line)
        for line in result.stdout.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    names = {requirement.name for requirement in requirements}
    assert required <= names
    assert not names & forbidden
    for requirement in requirements:
        if requirement.name == "numpy":
            assert Version(next(iter(requirement.specifier)).version).major == 1
        elif requirement.name == "opencv-python-headless":
            assert Version(next(iter(requirement.specifier)).version) < Version("4.12")


@pytest.mark.skipif(shutil.which("uv") is None, reason="backend conflict checks require uv")
@pytest.mark.parametrize(
    "first,second",
    tuple(
        combinations(
            (
                "onnx-cpu",
                "onnx-gpu-cu11",
                "onnx-gpu-cu12",
                "onnx-gpu-cu13",
                "ascend",
                "tensorrt-cu12",
            ),
            2,
        )
    ),
)
def test_inference_backend_extras_are_pairwise_exclusive(tmp_path, first, second):
    env = dict(os.environ, UV_CACHE_DIR=str(tmp_path / "uv-cache"), UV_OFFLINE="1")
    result = subprocess.run(
        ["uv", "export", "--frozen", "--extra", first, "--extra", second],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "conflict" in result.stderr.lower()


@pytest.mark.skipif(shutil.which("uv") is None, reason="extra validation requires uv")
def test_removed_generic_gpu_extra_is_rejected(tmp_path):
    env = dict(os.environ, UV_CACHE_DIR=str(tmp_path / "uv-cache"), UV_OFFLINE="1")
    result = subprocess.run(
        ["uv", "export", "--frozen", "--extra", "onnx-gpu"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "onnx-gpu" in result.stderr


def test_tensorrt_uses_official_wheels_and_platform_markers():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    requirements = [
        Requirement(raw) for raw in project["project"]["optional-dependencies"]["tensorrt-cu12"]
    ]
    for raw_environment, count in [
        (("linux", "x86_64", "3.10"), 4),
        (("win32", "AMD64", "3.12"), 4),
        (("darwin", "arm64", "3.12"), 0),
        (("linux", "aarch64", "3.12"), 0),
        (("linux", "x86_64", "3.13"), 0),
    ]:
        platform, machine, python = raw_environment
        environment = dict(
            sys_platform=platform,
            platform_machine=machine,
            python_version=python,
            python_full_version=python + ".0",
            implementation_name="cpython",
        )
        assert (
            sum(requirement.marker.evaluate(environment) for requirement in requirements) == count
        )
    for name in ("tensorrt-cu12-bindings", "tensorrt-cu12-libs"):
        package = next(p for p in lock["package"] if p["name"] == name)
        assert package["version"] == "10.16.0.72"
        assert package["source"] == {"registry": "https://pypi.nvidia.com/"}
        assert "sdist" not in package and package["wheels"]
        assert all("sha256:" in wheel["hash"] for wheel in package["wheels"])
