"""Guard failures that Twine's description checks do not detect."""

import importlib.util
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
