"""Validate tagged SDK sources and publish verified GitHub Release artifacts (Python 3.12)."""

import argparse
import ast
import hashlib
import json
import re
import subprocess
import tarfile
import tempfile
import zipfile
from pathlib import Path

import tomllib

TAG = re.compile(r"v(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)")


def version_for(tag):
    if not TAG.fullmatch(tag):
        raise ValueError("tag must be vMAJOR.MINOR.PATCH, without leading zeros")
    return tag[1:]


def notes_for(source, tag):
    version = version_for(tag)
    metadata = tomllib.loads((source / "pyproject.toml").read_text())
    tree = ast.parse((source / "src/ai_inference/__init__.py").read_text())
    versions = [
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "__version__" for t in node.targets)
    ]
    if metadata["project"]["version"] != version or versions != [version]:
        raise ValueError("tag, project version and public __version__ must agree")
    lines = (source / "CHANGELOG.md").read_text().splitlines(keepends=True)
    headings = [i for i, line in enumerate(lines) if line.startswith("## ")]
    matches = [i for i in headings if re.match(rf"^## {re.escape(version)}(?:\s|$)", lines[i])]
    if len(matches) != 1:
        raise ValueError("CHANGELOG must contain exactly one matching version heading")
    start = matches[0]
    end = next((i for i in headings if i > start), len(lines))
    body = "".join(lines[start + 1 : end]).strip()
    substantive = re.sub(r"<!--.*?-->", "", body, flags=re.S)
    substantive = re.sub(r"^#{1,6} .*?$", "", substantive, flags=re.M)
    if not substantive.strip():
        raise ValueError("CHANGELOG version section is empty")
    return "".join(lines[start:end]).strip() + "\n"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_members(names):
    for name in names:
        parts = Path(name).parts
        if (
            "__pycache__" in parts
            or "build" in parts
            or "models" in parts
            or name.endswith((".pyc", ".pyo", ".onnx", ".om", ".pt", ".pth"))
        ):
            raise ValueError(f"forbidden distribution member: {name}")


def prepare(source, tag, output):
    notes = notes_for(source, tag)
    version = version_for(tag)
    names = [f"ai_inference-{version}-py3-none-any.whl", f"ai_inference-{version}.tar.gz"]
    if sorted(p.name for p in (source / "dist").iterdir() if p.name != ".gitignore") != sorted(
        names
    ):
        raise ValueError("dist must contain only this version's fresh wheel and sdist")
    with zipfile.ZipFile(source / "dist" / names[0]) as wheel:
        check_members(wheel.namelist())
    with tarfile.open(source / "dist" / names[1]) as sdist:
        check_members(sdist.getnames())
    output.mkdir(parents=True, exist_ok=True)
    for name in names:
        (output / name).write_bytes((source / "dist" / name).read_bytes())
    sums = "".join(f"{digest(output / name)}  {name}\n" for name in sorted(names))
    (output / "SHA256SUMS").write_text(sums)
    (output / "notes.md").write_text(notes)
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True)
    (output / "release.json").write_text(
        json.dumps({"tag": tag, "commit": commit.strip(), "files": names + ["SHA256SUMS"]})
    )


def gh(*args):
    return subprocess.check_output(["gh", *args], text=True)


def publish(bundle, repo):
    data = json.loads((bundle / "release.json").read_text())
    tag = data["tag"]
    version = version_for(tag)
    names = [f"ai_inference-{version}-py3-none-any.whl", f"ai_inference-{version}.tar.gz"]
    if data["files"] != names + ["SHA256SUMS"]:
        raise ValueError("unexpected release files")
    expected = "".join(f"{digest(bundle / name)}  {name}\n" for name in sorted(names))
    if (bundle / "SHA256SUMS").read_text() != expected:
        raise ValueError("artifact checksum mismatch")
    ref = json.loads(gh("api", f"repos/{repo}/git/ref/tags/{tag}"))["object"]
    while ref["type"] == "tag":
        ref = json.loads(gh("api", f"repos/{repo}/git/tags/{ref['sha']}"))["object"]
    if ref["type"] != "commit" or ref["sha"] != data["commit"]:
        raise ValueError("remote tag moved since validation; refusing to publish")
    # The by-tag endpoint only promises published releases; list also includes drafts.
    pages = json.loads(gh("api", "--paginate", "--slurp", f"repos/{repo}/releases?per_page=100"))
    matches = [item for page in pages for item in page if item["tag_name"] == tag]
    if len(matches) > 1:
        raise ValueError("multiple releases match tag; inspect them manually")
    release = matches[0] if matches else None
    files = data["files"]
    if release and not release["draft"]:
        if sorted(a["name"] for a in release["assets"]) != sorted(files):
            raise ValueError("published asset set differs; refusing overwrite")
        with tempfile.TemporaryDirectory() as directory:
            gh("release", "download", tag, "--repo", repo, "--dir", directory)
            if any(digest(Path(directory) / name) != digest(bundle / name) for name in files):
                raise ValueError("published bytes differ; refusing overwrite")
        if (release["body"] or "").strip() != (bundle / "notes.md").read_text().strip():
            raise ValueError("published notes differ; refusing overwrite")
        print(f"{tag}: already published with identical notes and assets")
        return
    if release and any(a["name"] not in files for a in release["assets"]):
        raise ValueError("draft contains unexpected attachments; inspect it manually")
    if release is None:
        gh(
            "release",
            "create",
            tag,
            "--repo",
            repo,
            "--verify-tag",
            "--draft",
            "--title",
            f"ai-inference {version}",
            "--notes-file",
            str(bundle / "notes.md"),
        )
    gh(
        "release",
        "upload",
        tag,
        "--repo",
        repo,
        "--clobber",
        *(str(bundle / name) for name in files),
    )
    with tempfile.TemporaryDirectory() as directory:
        gh("release", "download", tag, "--repo", repo, "--dir", directory)
        if any(digest(Path(directory) / name) != digest(bundle / name) for name in files):
            raise ValueError("uploaded bytes differ; leaving release in draft")
    # Recheck the tag immediately before making the completed draft public.
    current = json.loads(gh("api", f"repos/{repo}/git/ref/tags/{tag}"))["object"]
    while current["type"] == "tag":
        current = json.loads(gh("api", f"repos/{repo}/git/tags/{current['sha']}"))["object"]
    if current["sha"] != data["commit"]:
        raise ValueError("tag moved; leaving release in draft")
    gh(
        "release",
        "edit",
        tag,
        "--repo",
        repo,
        "--draft=false",
        "--title",
        f"ai-inference {version}",
        "--notes-file",
        str(bundle / "notes.md"),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["tag", "validate", "prepare", "publish"])
    parser.add_argument("--tag")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repo")
    args = parser.parse_args()
    if args.command == "tag":
        version_for(args.tag)
        print(args.tag)
    elif args.command == "validate":
        print(notes_for(args.source, args.tag), end="")
    elif args.command == "prepare":
        prepare(args.source, args.tag, args.output)
    else:
        publish(args.output, args.repo)


if __name__ == "__main__":
    main()
