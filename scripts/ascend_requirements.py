"""Generate pip Ascend requirements from the extra's sources and locked wheel hashes."""

import argparse
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 development environments install tomli.
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
HEADER = """# Generated from pyproject.toml and uv.lock by scripts/ascend_requirements.py; do not edit.
# Ascend vendor runtimes: Linux x86_64/aarch64, CPython 3.10-3.12.
# Install with the SDK's ascend extra; CANN and drivers are installed separately.
# URL SHA256 fragments pin the vendor builds. Vendor URLs stay out of package metadata.
"""


def render(project: dict, lock: dict) -> str:
    hashes = {
        wheel["url"]: wheel["hash"]
        for package in lock["package"]
        for wheel in package.get("wheels", [])
    }
    requirements = []
    for dependency in project["project"]["optional-dependencies"]["ascend"]:
        name, separator, _ = dependency.partition("==")
        if not separator:
            raise ValueError("Ascend dependencies must declare pinned package versions")
        for source in project["tool"]["uv"]["sources"][name.strip()]:
            if source.get("extra") != "ascend":
                continue
            digest = hashes[source["url"]]
            if not digest.startswith("sha256:"):
                raise ValueError("Ascend wheels must have locked SHA256 hashes")
            url = source["url"] + "#sha256=" + digest.removeprefix("sha256:")
            requirements.append(f"{name.strip()} @ {url} ; {source['marker']}")
    if not requirements:
        raise ValueError("ascend must have pinned uv wheel sources")
    return HEADER + "\n".join(requirements) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="check without changing the file")
    args = parser.parse_args()
    expected = render(
        tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8")),
        tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8")),
    )
    output = ROOT / "requirements/ascend.txt"
    if args.check:
        if not output.exists() or output.read_text(encoding="utf-8") != expected:
            parser.exit(
                1, "Ascend requirements are stale; run python scripts/ascend_requirements.py\n"
            )
        print("Ascend requirements match the extra's pinned wheel sources")
    else:
        output.write_text(expected, encoding="utf-8")
        print("Generated requirements/ascend.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
