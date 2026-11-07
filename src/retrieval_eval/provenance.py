"""Run provenance: hardware/OS, exact installed dependency versions, code identity, file hashes.

Anything that cannot be determined is reported as the string "unknown" (never guessed).
Standard library only (no psutil/threadpoolctl).
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import subprocess
import sys
from pathlib import Path

UNKNOWN = "unknown"
THREAD_ENV_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _cpu_model() -> str:
    try:
        if sys.platform.startswith("linux"):
            with open("/proc/cpuinfo", encoding="utf-8") as fh:
                for line in fh:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        if sys.platform == "win32":
            ident = os.environ.get("PROCESSOR_IDENTIFIER")
            if ident:
                return ident
        if sys.platform == "darwin":
            out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                 capture_output=True, text=True, timeout=5)
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip()
    except Exception:  # noqa: BLE001 - provenance must never break a run
        pass
    return UNKNOWN


def _total_ram_bytes():
    try:
        if sys.platform.startswith("linux"):
            with open("/proc/meminfo", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) * 1024  # value is in kB
        if sys.platform == "win32":
            import ctypes

            class _MemStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = _MemStatus()
            status.dwLength = ctypes.sizeof(_MemStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullTotalPhys)
        if sys.platform == "darwin":
            out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5)
            if out.returncode == 0:
                return int(out.stdout.strip())
    except Exception:  # noqa: BLE001
        pass
    return UNKNOWN


def collect_environment() -> dict:
    torch = sys.modules.get("torch")
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "cpu_model": _cpu_model(),
        "logical_cpus": os.cpu_count() or UNKNOWN,
        "total_ram_bytes": _total_ram_bytes(),
        "gpu": "not used (device is cpu); GPU inventory not collected",
        "thread_env": {name: os.environ.get(name) for name in THREAD_ENV_VARS},
        "torch_threads_observed": torch.get_num_threads() if torch is not None else None,
    }


def collect_dependencies() -> dict:
    """Exact installed version of every distribution in the running environment (name -> version)."""
    found = {}
    for dist in importlib.metadata.distributions():
        name = dist.metadata["Name"]
        if name:
            found[name.lower().replace("_", "-")] = dist.version
    return dict(sorted(found.items()))


def _git(root: Path, *args):
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, timeout=15)


def git_state(root: Path) -> dict:
    """Commit and dirty state, only if `root` is itself the top level of a git repository with a commit."""
    try:
        top = _git(root, "rev-parse", "--show-toplevel")
        if top.returncode != 0:
            return {"available": False, "reason": "not a git repository"}
        if Path(top.stdout.strip()).resolve() != root.resolve():
            return {"available": False, "reason": "project root is not the top level of a git repository"}
        head = _git(root, "rev-parse", "HEAD")
        if head.returncode != 0:
            return {"available": False, "reason": "repository has no commits"}
        status = _git(root, "status", "--porcelain")
        return {"available": True, "commit": head.stdout.strip(), "dirty": bool(status.stdout.strip())}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "reason": f"git unavailable ({type(exc).__name__})"}


def source_fingerprint(package_dir: Path, root: Path | None = None) -> dict:
    """SHA-256 over the sorted (path, file hash) list of the package's .py files plus pyproject.toml/uv.lock
    when present. Paths are location-independent ("retrieval_eval/x.py")."""
    entries = []
    for p in sorted(package_dir.rglob("*.py")):
        entries.append((f"{package_dir.name}/{p.relative_to(package_dir).as_posix()}", sha256_file(p)))
    if root is not None:
        for name in ("pyproject.toml", "uv.lock"):
            if (root / name).is_file():
                entries.append((name, sha256_file(root / name)))
    entries.sort()
    digest = hashlib.sha256("\n".join(f"{n} {h}" for n, h in entries).encode("utf-8")).hexdigest()
    return {"sha256": digest, "files": [{"path": n, "sha256": h} for n, h in entries]}


def collect_code_identity(package_dir: Path | None = None, root: Path | None = None) -> dict:
    package_dir = package_dir or Path(__file__).resolve().parent
    root = root or package_dir.parents[1]
    if not (root / "pyproject.toml").is_file():
        root_for_git, root_for_files = None, None  # installed copy; no project checkout beside it
    else:
        root_for_git, root_for_files = root, root
    return {
        "git": git_state(root_for_git) if root_for_git else {"available": False, "reason": "no project checkout found"},
        "source_fingerprint": source_fingerprint(package_dir, root_for_files),
    }
