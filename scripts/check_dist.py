"""Check PyPI distribution metadata, then install the wheel in clean environments.

Requires uv on PATH. This command never uploads artifacts.
"""

import argparse
import email
import os
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path


def check_metadata(raw, members, license_path):
    metadata = email.message_from_bytes(raw)
    if metadata.get("License-Expression") != "MIT":
        raise ValueError("distribution must declare the MIT license")
    if "LICENSE" not in metadata.get_all("License-File", []) or license_path not in members:
        raise ValueError("distribution must include LICENSE")
    for requirement in metadata.get_all("Requires-Dist", []):
        if "@" in requirement:
            raise ValueError(f"PyPI rejects direct URL dependencies: {requirement}")
    return metadata


def check_artifacts(directory):
    wheels = list(directory.glob("*.whl"))
    sources = list(directory.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sources) != 1:
        raise ValueError("expected exactly one wheel and one source distribution")
    wheel, source = wheels[0], sources[0]
    with zipfile.ZipFile(wheel) as archive:
        members = archive.namelist()
        names = [name for name in members if name.endswith(".dist-info/METADATA")]
        if len(names) != 1:
            raise ValueError("wheel must contain exactly one METADATA")
        metadata = check_metadata(
            archive.read(names[0]), members, names[0].rsplit("/", 1)[0] + "/licenses/LICENSE"
        )
        if "ai_inference/py.typed" not in members:
            raise ValueError("wheel must include py.typed")
    with tarfile.open(source) as archive:
        root = source.name.removesuffix(".tar.gz")
        stream = archive.extractfile(root + "/PKG-INFO")
        if stream is None:
            raise ValueError("sdist must contain PKG-INFO")
        source_metadata = check_metadata(stream.read(), archive.getnames(), root + "/LICENSE")
        if root + "/requirements/ascend.txt" not in archive.getnames():
            raise ValueError("sdist must include the standalone Ascend requirements")
    for field in ("Name", "Version", "Requires-Python", "Requires-Dist", "Provides-Extra"):
        if metadata.get_all(field) != source_metadata.get_all(field):
            raise ValueError(f"wheel and sdist disagree on {field}")
    for path in (wheel, source):
        expected = f"ai_inference-{metadata['Version']}"
        if not path.name.startswith(expected + ("-" if path == wheel else ".")):
            raise ValueError("artifact filename and metadata version disagree")
    return wheel, metadata["Version"]


def check_install(wheel, version):
    # -I and an unrelated cwd ensure imports cannot fall back to repository sources.
    base = """
import importlib.metadata
import sys
from pathlib import Path
import ai_inference
assert Path(ai_inference.__file__).is_relative_to(Path(sys.prefix))
assert ai_inference.__version__ == importlib.metadata.version('ai-inference') == sys.argv[1]
assert all(hasattr(ai_inference, name) for name in ai_inference.__all__)
"""
    with tempfile.TemporaryDirectory(prefix="ai-inference-install-") as directory:
        root = Path(directory)
        for name, extra in (("base", ""), ("service-cpu", "[service,onnx-cpu]")):
            env = root / name
            subprocess.run(["uv", "venv", "--python", sys.executable, str(env)], check=True)
            python = env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            subprocess.run(
                ["uv", "pip", "install", "--python", str(python), str(wheel) + extra], check=True
            )
            subprocess.run(["uv", "pip", "check", "--python", str(python)], check=True)
            code = base
            if extra:
                code += "\nimport onnxruntime\nfrom ai_inference.service.app import app\n"
                code += "assert app.openapi()['paths']\n"
            else:
                code += "\nimport importlib.util\n"
                code += "assert importlib.util.find_spec('fastapi') is None\n"
                code += "assert importlib.util.find_spec('onnxruntime') is None\n"
            subprocess.run([str(python), "-I", "-c", code, version], cwd=root, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    wheel, version = check_artifacts(args.directory.resolve())
    check_install(wheel, version)
    print(f"Distribution metadata and clean installs passed: {version}")


if __name__ == "__main__":
    main()
