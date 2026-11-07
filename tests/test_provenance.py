import importlib.metadata
import shutil
import subprocess
from pathlib import Path

import pytest

from retrieval_eval import provenance


def test_environment_has_required_fields_and_unknown_is_never_invented(monkeypatch):
    env = provenance.collect_environment()
    for key in ("python_version", "platform", "machine", "cpu_model", "logical_cpus", "total_ram_bytes", "thread_env"):
        assert key in env
    assert env["python_version"].count(".") >= 1
    # simulate unreadable hardware info: must say "unknown", not raise and not guess
    real_open = open

    def deny(path, *a, **k):
        if str(path) in ("/proc/cpuinfo", "/proc/meminfo"):
            raise OSError("denied")
        return real_open(path, *a, **k)

    monkeypatch.setattr("builtins.open", deny)
    monkeypatch.setattr(provenance.sys, "platform", "freebsd")
    assert provenance._cpu_model() == "unknown" and provenance._total_ram_bytes() == "unknown"


def test_dependencies_are_exact_installed_versions():
    deps = provenance.collect_dependencies()
    assert deps["pytest"] == importlib.metadata.version("pytest")
    assert deps["numpy"] == importlib.metadata.version("numpy") and deps["rank-bm25"] == importlib.metadata.version("rank-bm25")
    assert list(deps) == sorted(deps)


def test_sha256_file(tmp_path):
    p = tmp_path / "f"
    p.write_bytes(b"abc")
    assert provenance.sha256_file(p) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_source_fingerprint_is_deterministic_and_sensitive(tmp_path):
    pkg = tmp_path / "pkg" / "retrieval_eval"
    pkg.mkdir(parents=True)
    (pkg / "a.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "pkg" / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    f1 = provenance.source_fingerprint(pkg, tmp_path / "pkg")
    assert f1 == provenance.source_fingerprint(pkg, tmp_path / "pkg")
    assert [e["path"] for e in f1["files"]] == ["pyproject.toml", "retrieval_eval/a.py"]  # location-independent paths
    (pkg / "a.py").write_text("x = 2\n", encoding="utf-8")
    assert provenance.source_fingerprint(pkg, tmp_path / "pkg")["sha256"] != f1["sha256"]


def test_real_package_has_a_code_identity():
    ident = provenance.collect_code_identity()
    assert ident["source_fingerprint"]["sha256"] and {"git", "source_fingerprint"} <= set(ident)
    assert any(f["path"].endswith("runner.py") for f in ident["source_fingerprint"]["files"])


def test_git_state_not_a_repository(tmp_path):
    assert provenance.git_state(tmp_path)["available"] is False


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_git_state_commit_dirty_and_nested_directory(tmp_path):
    def git(*args):
        subprocess.run(["git", "-C", str(tmp_path), "-c", "user.email=t@t", "-c", "user.name=t", *args],
                       check=True, capture_output=True)

    git("init")
    assert provenance.git_state(tmp_path) == {"available": False, "reason": "repository has no commits"}
    (tmp_path / "f.txt").write_text("1", encoding="utf-8")
    git("add", ".")
    git("commit", "-m", "c")
    clean = provenance.git_state(tmp_path)
    assert clean["available"] and len(clean["commit"]) == 40 and clean["dirty"] is False
    (tmp_path / "f.txt").write_text("2", encoding="utf-8")
    assert provenance.git_state(tmp_path)["dirty"] is True
    sub = tmp_path / "sub"
    sub.mkdir()  # a project nested inside an unrelated repo must not borrow that repo's commit
    state = provenance.git_state(sub)
    assert state["available"] is False and "top level" in state["reason"]
