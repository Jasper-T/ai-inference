"""Guard failures that Twine's description checks do not detect."""

import importlib.util
import io
import tarfile
import zipfile
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "check_dist", Path(__file__).resolve().parents[1] / "scripts/check_dist.py"
)
check_dist = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_dist)


def test_distribution_rejects_optional_direct_url():
    metadata = b"""Metadata-Version: 2.4
License-Expression: MIT
License-File: LICENSE
Requires-Dist: vendor@ https://example.com/vendor.whl ; extra == "ascend"

"""
    with pytest.raises(ValueError, match="direct URL"):
        check_dist.check_metadata(metadata, ["LICENSE"], "LICENSE")


def test_distribution_requires_bundled_license():
    metadata = b"License-Expression: MIT\nLicense-File: LICENSE\n\n"
    with pytest.raises(ValueError, match="include LICENSE"):
        check_dist.check_metadata(metadata, [], "LICENSE")


def test_distribution_accepts_index_dependencies():
    metadata = b"""License-Expression: MIT
License-File: LICENSE
Requires-Dist: onnxruntime>=1.18,<2 ; extra == "onnx-cpu"

"""
    check_dist.check_metadata(metadata, ["LICENSE"], "LICENSE")


@pytest.mark.parametrize(
    "missing",
    [
        None,
        ".github/workflows/package.yml",
        ".github/workflows/release.yml",
        ".github/build-constraints.txt",
    ],
)
def test_sdist_contains_files_needed_for_release_tests_and_builds(tmp_path, missing):
    version = "0.2.3"
    prefix = f"ai_inference-{version}"
    metadata = (
        f"Name: ai-inference\nVersion: {version}\nRequires-Python: >=3.10\n"
        "License-Expression: MIT\nLicense-File: LICENSE\n\n"
    ).encode()
    with zipfile.ZipFile(tmp_path / f"{prefix}-py3-none-any.whl", "w") as wheel:
        wheel.writestr(f"{prefix}.dist-info/METADATA", metadata)
        wheel.writestr(f"{prefix}.dist-info/licenses/LICENSE", "MIT")
        wheel.writestr("ai_inference/py.typed", "")
    members = {
        "PKG-INFO": metadata,
        "LICENSE": b"MIT",
        "requirements/ascend.txt": b"",
        ".github/workflows/package.yml": b"jobs: {}",
        ".github/workflows/release.yml": b"jobs: {}",
        ".github/build-constraints.txt": b"setuptools==80.9.0",
    }
    with tarfile.open(tmp_path / f"{prefix}.tar.gz", "w:gz") as source:
        for name, data in members.items():
            if name == missing:
                continue
            info = tarfile.TarInfo(f"{prefix}/{name}")
            info.size = len(data)
            source.addfile(info, io.BytesIO(data))
    if missing:
        with pytest.raises(ValueError, match="sdist must include") as error:
            check_dist.check_artifacts(tmp_path)
        assert missing in str(error.value)
    else:
        assert check_dist.check_artifacts(tmp_path)[1] == version
