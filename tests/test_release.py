"""Exercise release failure guards without contacting GitHub or publishing anything."""

import importlib.util
import json
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(sys.version_info < (3, 11), reason="release tooling uses tomllib")
if sys.version_info >= (3, 11):
    spec = importlib.util.spec_from_file_location(
        "release_tool", Path(__file__).resolve().parents[1] / "scripts/release.py"
    )
    release = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(release)


@pytest.fixture
def source(tmp_path):
    (tmp_path / "src/ai_inference").mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "0.2.0"\n')
    (tmp_path / "src/ai_inference/__init__.py").write_text('__version__ = "0.2.0"\n')
    (tmp_path / "CHANGELOG.md").write_text(
        "## Unreleased\nfuture\n## 0.2.0 — today\n\n### Added\n- SDK\n\n## Other\nexcluded\n"
    )
    return tmp_path


def test_exact_changelog_section(source):
    assert release.notes_for(source, "v0.2.0") == "## 0.2.0 — today\n\n### Added\n- SDK\n"


@pytest.mark.parametrize("tag", ["0.2.0", "v01.2.0", "v0.2.0;echo x", "v0.2.0\n", "v0.2"])
def test_bad_tag(tag):
    with pytest.raises(ValueError):
        release.version_for(tag)


@pytest.mark.parametrize(
    "body",
    ["## 0.2.1\nwrong", "## 0.2.0\n", "## 0.2.0\n<!-- pending -->", "## 0.2.0\nx\n## 0.2.0\ny"],
)
def test_missing_empty_duplicate_notes(source, body):
    (source / "CHANGELOG.md").write_text(body)
    with pytest.raises(ValueError):
        release.notes_for(source, "v0.2.0")


@pytest.mark.parametrize(
    "file,content",
    [
        ("pyproject.toml", '[project]\nversion="0.2.1"'),
        ("src/ai_inference/__init__.py", '__version__ = "0.2.1"'),
    ],
)
def test_version_mismatch(source, file, content):
    (source / file).write_text(content)
    with pytest.raises(ValueError):
        release.notes_for(source, "v0.2.0")


@pytest.mark.parametrize(
    "name",
    [
        "pkg/build/lib/ai_inference/service/app.py",
        "pkg/models/model.om",
        "pkg/__pycache__/x.pyc",
    ],
)
def test_forbidden_artifact_content(name):
    with pytest.raises(ValueError):
        release.check_members([name])


def test_service_sources_docs_and_tests_are_allowed_in_distribution():
    release.check_members(
        [
            "ai_inference/service/app.py",
            "pkg/src/ai_inference/service/app.py",
            "pkg/docs/service/README.md",
            "pkg/tests/test_service.py",
        ]
    )


@pytest.fixture
def bundle(tmp_path):
    names = ["ai_inference-0.2.0-py3-none-any.whl", "ai_inference-0.2.0.tar.gz"]
    for name in names:
        (tmp_path / name).write_bytes(name.encode())
    (tmp_path / "SHA256SUMS").write_text(
        "".join(f"{release.digest(tmp_path / name)}  {name}\n" for name in sorted(names))
    )
    (tmp_path / "notes.md").write_text("## 0.2.0\nSDK\n")
    (tmp_path / "release.json").write_text(
        json.dumps({"tag": "v0.2.0", "commit": "a" * 40, "files": names + ["SHA256SUMS"]})
    )
    return tmp_path


def fake_github(
    monkeypatch, bundle, *, draft=False, changed=False, moved=False, error=None, missing=False
):
    calls = []
    files = json.loads((bundle / "release.json").read_text())["files"]

    def gh(*args):
        calls.append(args)
        if args[0] == "api" and "--paginate" in args:
            if error:
                raise RuntimeError(error)
            return json.dumps(
                [
                    []
                    if missing
                    else [
                        {
                            "tag_name": "v0.2.0",
                            "draft": draft,
                            "body": (bundle / "notes.md").read_text(),
                            "assets": [{"name": name} for name in files],
                        }
                    ]
                ]
            )
        if args[0] == "api":
            return json.dumps({"object": {"type": "commit", "sha": ("b" if moved else "a") * 40}})
        if args[:2] == ("release", "download"):
            directory = Path(args[args.index("--dir") + 1])
            for name in files:
                (directory / name).write_bytes(
                    b"different" if changed else (bundle / name).read_bytes()
                )
        return ""

    monkeypatch.setattr(release, "gh", gh)
    return calls


def test_identical_published_release_is_read_only(monkeypatch, bundle):
    calls = fake_github(monkeypatch, bundle)
    release.publish(bundle, "owner/repo")
    assert not any(c[:2] in [("release", "upload"), ("release", "edit")] for c in calls)


def test_published_different_bytes_fail_before_mutation(monkeypatch, bundle):
    calls = fake_github(monkeypatch, bundle, changed=True)
    with pytest.raises(ValueError, match="bytes differ"):
        release.publish(bundle, "owner/repo")
    assert not any(c[:2] == ("release", "upload") for c in calls)


def test_matching_draft_uploads_before_publishing(monkeypatch, bundle):
    calls = fake_github(monkeypatch, bundle, draft=True)
    release.publish(bundle, "owner/repo")
    commands = [c[:2] for c in calls]
    assert (
        commands.index(("release", "upload"))
        < commands.index(("release", "download"))
        < commands.index(("release", "edit"))
    )


def test_moved_tag_fails_before_mutation(monkeypatch, bundle):
    calls = fake_github(monkeypatch, bundle, moved=True)
    with pytest.raises(ValueError, match="tag moved"):
        release.publish(bundle, "owner/repo")
    assert len(calls) == 1


def test_auth_failure_does_not_create_release(monkeypatch, bundle):
    calls = fake_github(monkeypatch, bundle, error="HTTP 403 forbidden")
    with pytest.raises(RuntimeError):
        release.publish(bundle, "owner/repo")
    assert not any(c[:2] == ("release", "create") for c in calls)


def test_corrupt_bundle_fails_before_api(monkeypatch, bundle):
    calls = fake_github(monkeypatch, bundle)
    (bundle / "SHA256SUMS").write_text("bad")
    with pytest.raises(ValueError, match="checksum"):
        release.publish(bundle, "owner/repo")
    assert calls == []


def test_new_draft_creation_precedes_upload(monkeypatch, bundle):
    calls = fake_github(monkeypatch, bundle, missing=True)
    release.publish(bundle, "owner/repo")
    commands = [c[:2] for c in calls]
    assert commands.index(("release", "create")) < commands.index(("release", "upload"))
    assert commands[-1] == ("release", "edit")


def test_bad_uploaded_bytes_keep_draft(monkeypatch, bundle):
    calls = fake_github(monkeypatch, bundle, draft=True, changed=True)
    with pytest.raises(ValueError, match="leaving release in draft"):
        release.publish(bundle, "owner/repo")
    assert not any(c[:2] == ("release", "edit") for c in calls)


@pytest.mark.parametrize("python", ["3.10", "3.11", "3.12"])
@pytest.mark.parametrize("arch", ["aarch64", "x86_64"])
def test_ascend_wheel_selection(python, arch):
    from packaging.requirements import Requirement

    root = Path(__file__).resolve().parents[1]
    extras = release.tomllib.loads((root / "pyproject.toml").read_text())["project"][
        "optional-dependencies"
    ]
    assert set(extras) == {"service", "onnx-cpu", "onnx-gpu", "ascend"}
    environment = {
        "sys_platform": "linux",
        "platform_machine": arch,
        "python_version": python,
        "implementation_name": "cpython",
    }
    selected = [
        Requirement(dep)
        for dep in extras["ascend"]
        if Requirement(dep).marker.evaluate(environment)
    ]
    assert sorted(dep.name for dep in selected) == ["aclruntime", "ais-bench"]
    acl = next(dep for dep in selected if dep.name == "aclruntime")
    cp = "cp" + python.replace(".", "")
    assert acl.url.endswith(f"-{cp}-{cp}-linux_{arch}.whl")


@pytest.mark.parametrize(
    "python,arch,platform,implementation",
    [
        ("3.13", "aarch64", "linux", "cpython"),
        ("3.12", "arm64", "darwin", "cpython"),
        ("3.12", "x86_64", "win32", "cpython"),
        ("3.10", "aarch64", "linux", "pypy"),
    ],
)
def test_ascend_does_not_select_incompatible_wheels(python, arch, platform, implementation):
    from packaging.requirements import Requirement

    root = Path(__file__).resolve().parents[1]
    deps = release.tomllib.loads((root / "pyproject.toml").read_text())["project"][
        "optional-dependencies"
    ]["ascend"]
    environment = {
        "sys_platform": platform,
        "platform_machine": arch,
        "python_version": python,
        "implementation_name": implementation,
    }
    assert not any(Requirement(dep).marker.evaluate(environment) for dep in deps)


def test_prepare_accepts_uv_gitignore_and_checksums(source, monkeypatch):
    dist = source / "dist"
    dist.mkdir()
    (dist / ".gitignore").write_text("*")
    with zipfile.ZipFile(dist / "ai_inference-0.2.0-py3-none-any.whl", "w") as archive:
        archive.writestr("ai_inference/__init__.py", '__version__="0.2.0"')
    with tarfile.open(dist / "ai_inference-0.2.0.tar.gz", "w:gz"):
        pass
    monkeypatch.setattr(release.subprocess, "check_output", lambda *a, **k: "a" * 40)
    output = source / "bundle"
    release.prepare(source, "v0.2.0", output)
    data = json.loads((output / "release.json").read_text())
    assert data["commit"] == "a" * 40
    assert len((output / "SHA256SUMS").read_text().splitlines()) == 2
    assert sorted(p.name for p in output.iterdir()) == sorted(
        data["files"] + ["notes.md", "release.json"]
    )


def test_prepare_rejects_stale_dist(source):
    dist = source / "dist"
    dist.mkdir()
    (dist / "old.whl").write_bytes(b"old")
    with pytest.raises(ValueError, match="fresh wheel"):
        release.prepare(source, "v0.2.0", source / "bundle")
