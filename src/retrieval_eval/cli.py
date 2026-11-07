"""Command-line entry point:  python -m retrieval_eval.cli --out-dir runs/demo   (see README).

Exit codes: 0 = completed run; 1 = run failed or export failed (incomplete; ``run_failed.json`` written);
2 = invalid configuration/input, rejected before any model loading (nothing written).

This module imports no numpy/torch at import time, so thread environment variables can be set first.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DATASET = "data/synthetic"
DATASET_FILES = ("documents.jsonl", "queries.jsonl", "qrels.csv")
FIXTURE_MANIFEST = "FIXTURE.sha256"
THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
LOCK_NAME = ".retrieval_eval.lock"  # exclusive reservation of an output directory for ONE invocation


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m retrieval_eval.cli",
        description="Run BM25 and the embedding retriever over one frozen dataset on CPU; write quality metrics, "
                    "measured retrieval latency and reproducible JSON/CSV artifacts.")
    p.add_argument("--dataset", default=DEFAULT_DATASET, help="directory with documents.jsonl, queries.jsonl, qrels.csv "
                   f"(default: {DEFAULT_DATASET})")
    p.add_argument("--out-dir", required=True, help="output directory; must not exist or must be empty "
                   "(existing outputs are never overwritten)")
    p.add_argument("--k", default="1,3,5", help="comma-separated cutoffs, positive integers (default: 1,3,5)")
    p.add_argument("--warmup", type=int, default=3, help="discarded warm-up searches per query and method (default: 3)")
    p.add_argument("--repetitions", type=int, default=20,
                   help="measured searches per query and method (default: 20)")
    p.add_argument("--threads", type=int, default=1, help="CPU threads for torch/BLAS (default: 1)")
    p.add_argument("--dataset-designation", choices=("synthetic", "external"), default="synthetic",
                   help="recorded in the manifest (default: synthetic)")
    return p


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load_dataset(config, runner, provenance):
    from .data import load_dataset

    d = config.dataset_dir
    if not d.is_dir():
        raise runner.ConfigError(f"dataset directory not found: {d}")
    missing = [f for f in DATASET_FILES if not (d / f).is_file()]
    if missing:
        raise runner.ConfigError(f"dataset directory {d} is missing: {', '.join(missing)}")
    dataset = load_dataset(d)  # DataValidationError (a ValueError) on malformed input
    files = [{"name": f, "sha256": provenance.sha256_file(d / f), "bytes": (d / f).stat().st_size} for f in DATASET_FILES]
    manifest = {"name": FIXTURE_MANIFEST, "present": False, "verified": None}
    mpath = d / FIXTURE_MANIFEST
    if mpath.is_file():
        listed = {}
        for line in mpath.read_text(encoding="utf-8").splitlines():
            if line.strip():
                digest, _, name = line.partition("  ")
                listed[name.strip()] = digest.strip()
        bad = [f["name"] for f in files if listed.get(f["name"]) != f["sha256"]]
        if bad:
            raise runner.ConfigError(f"dataset files do not match {FIXTURE_MANIFEST}: {', '.join(bad)} "
                                     "(the frozen fixture was modified)")
        manifest.update(present=True, verified=True)
    return dataset, {"directory": str(d), "files": files, "fixture_manifest": manifest}


class ReservationError(RuntimeError):
    """The output directory could not be reserved exclusively for this invocation."""


def _check_out_dir(path: Path, runner) -> Path:
    """Cheap early check (before any model work). It is advisory only: ownership is decided by OutputReservation."""
    if path.exists() and not path.is_dir():
        raise runner.ConfigError(f"output path exists and is not a directory: {path}")
    if path.is_dir():
        entries = sorted(p.name for p in path.iterdir())
        if entries == [LOCK_NAME]:
            raise runner.ConfigError(
                f"output directory is reserved by another invocation: {path} (lock file {LOCK_NAME}; if no run is "
                "active this is a stale lock from an interrupted run: delete it and retry)")
        if entries:
            raise runner.ConfigError(f"output directory is not empty: {path} (existing outputs are never "
                                     "overwritten; choose a new directory)")
    return path


class OutputReservation:
    """Exclusive ownership of an output directory for one invocation, held from before the first build until all
    artifacts (or the failure manifest) are written.

    The directory is created if needed (an existing EMPTY directory is supported). Ownership is the atomic creation
    (O_CREAT | O_EXCL) of ``.retrieval_eval.lock``: of any number of concurrent invocations exactly one succeeds. While
    holding it the directory contents are checked again, because a previous owner may have finished (and released)
    between the caller's early emptiness check and this attempt; anything besides the lock file means the directory is
    not ours and the reservation is released at once, leaving the other files untouched. Only the owner (matched by a
    random token) ever removes the lock. A lock left by a killed process is NOT removed automatically (it cannot be
    told from a live owner); the error message says how to proceed. Relies on O_EXCL being atomic on a local file system.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.lock = self.path / LOCK_NAME
        self.token = uuid.uuid4().hex
        self.held = False

    def acquire(self, command=None) -> "OutputReservation":
        try:
            self.path.mkdir(parents=True, exist_ok=True)
            fd = os.open(self.lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            raise ReservationError(
                f"output directory is reserved by another invocation: {self.path} (lock file {LOCK_NAME}; if no run "
                "is active this is a stale lock from an interrupted run: delete it and retry)") from None
        except OSError as exc:
            raise ReservationError(f"cannot reserve output directory {self.path}: {exc}") from exc
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump({"token": self.token, "pid": os.getpid(), "reserved_utc": _utc_now(),
                           "command": list(command or [])}, fh)
        except OSError as exc:
            self.held = True  # we do own the file we created; remove it
            self.release()
            raise ReservationError(f"cannot write the reservation in {self.path}: {exc}") from exc
        self.held = True
        others = sorted(p.name for p in self.path.iterdir() if p.name != LOCK_NAME)
        if others:  # a previous owner finished between the early check and this reservation
            self.release()
            raise ReservationError(f"output directory is not empty: {self.path} (existing outputs are never "
                                   "overwritten; choose a new directory)")
        return self

    def release(self) -> None:
        if not self.held:
            return
        self.held = False
        try:
            if json.loads(self.lock.read_text(encoding="utf-8")).get("token") == self.token:
                self.lock.unlink()
        except (OSError, ValueError):
            pass  # already gone or not ours: never remove a lock we cannot prove we own


def _print_summary(run: dict, out_dir: Path, written) -> None:
    ds = run["dataset"]
    first = run["results"][run["config"]["methods"][0]]["counts"]
    print(f"status: {run['run']['status']}   run_id: {run['run']['run_id']}")
    print(f"dataset: {ds['designation']} ({ds['counts']['documents']} documents, {ds['counts']['queries']} queries: "
          f"{first['scored']} scored, {first['no_judged_relevant']} with no judged-relevant document)")
    print(f"quality (macro mean over the {first['scored']} scored queries; synthetic demo observations):")
    print(f"  {'method':<10}{'k':>3}{'Recall':>9}{'MRR':>9}")
    for m in run["config"]["methods"]:
        for k in run["config"]["ks"]:
            a = run["results"][m]["aggregate"][str(k)]
            fmt = lambda v: "   n/a" if v is None else f"{v:9.4f}"  # noqa: E731
            print(f"  {m:<10}{k:>3}{fmt(a['macro_recall'])}{fmt(a['mrr'])}")
    print("latency (warm, local, serial microbenchmark; pooled over all queries; milliseconds):")
    for m in run["config"]["methods"]:
        o = run["latency"]["summary"][m]["overall"]
        print(f"  {m:<10} n={o['n']}  median={o['median_ns'] / 1e6:.3f} ms  p95={o['p95_ns'] / 1e6:.3f} ms")
    print(f"wrote {len(written)} files to {out_dir}: " + ", ".join(p.name for p in written))


def main(argv=None, *, builders=None, clock=None, _before_reserve=None) -> int:
    """Run the CLI. ``builders``/``clock``/``_before_reserve`` are test hooks (fake adapters, controlled clock, a callback that
    runs after the early output check and before the reservation); the defaults are the real thing."""
    argv_list = list(sys.argv[1:] if argv is None else argv)
    try:
        args = build_parser().parse_args(argv_list)
    except SystemExit as exc:  # argparse: usage errors exit 2, --help exits 0
        return exc.code if isinstance(exc.code, int) else 2

    from . import export, provenance, runner

    # ---- 1. validate everything cheap BEFORE any model loading or file creation
    try:
        config = runner.make_config(dataset_dir=args.dataset, ks=runner.parse_ks(args.k), warmup=args.warmup,
                                    repetitions=args.repetitions, threads=args.threads,
                                    designation=args.dataset_designation)
        dataset, dataset_info = _load_dataset(config, runner, provenance)
        out_dir = _check_out_dir(Path(args.out_dir), runner)
    except (runner.ConfigError, ValueError) as exc:  # includes DataValidationError
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # ---- 2. take exclusive ownership of the output directory BEFORE any build; released in the finally below
    started = _utc_now()
    command = ["python", "-m", "retrieval_eval.cli", *argv_list]
    if _before_reserve is not None:
        _before_reserve()
    reservation = OutputReservation(out_dir)
    try:
        reservation.acquire(command)
    except ReservationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    try:
        # ---- 3. fix thread settings before numpy/torch are imported (effective for BLAS/OpenMP on a fresh process)
        for name in THREAD_ENV:
            os.environ[name] = str(config.threads)

        def fail(errors: list[dict], partial: dict, run_id=None) -> int:
            doc = {
                "schema_version": runner.SCHEMA_VERSION, "status": "incomplete",
                "note": "INCOMPLETE RUN: no aggregate metrics are published and no run.json exists. Partial records are "
                        "progress information, not results.",
                "run_id": run_id, "started_utc": started, "failed_utc": _utc_now(), "command": command,
                "config": config.to_dict(), "dataset": dataset_info, "errors": errors, "partial": partial,
            }
            for e in errors:
                print(f"error: run incomplete at {e['stage']}"
                      + (f" (method={e['method']})" if e.get("method") else "")
                      + (f" (query={e['query_id']})" if e.get("query_id") else "")
                      + f": {e['error_type']}: {e['message']}", file=sys.stderr)
            try:
                path = export.write_failure_manifest(out_dir, doc)
                print(f"failure manifest: {path}", file=sys.stderr)
            except Exception as exc:  # noqa: BLE001
                print(f"error: could not write failure manifest: {exc}", file=sys.stderr)
            return 1

        # ---- 4. run, then export
        try:
            chosen = builders if builders is not None else runner.default_builders(dataset, config)
            run_doc = runner.run_evaluation(config, dataset, chosen, clock=clock or time.perf_counter_ns,
                                            command=command, dataset_info=dataset_info)
        except runner.RunStageError as err:
            return fail([err.to_dict()], err.partial)
        try:
            written = export.write_run_artifacts(out_dir, run_doc)
        except export.ExportError as err:
            return fail([{"stage": "export", "method": None, "query_id": None, "error_type": type(err).__name__,
                          "message": str(err), "traceback": ""}],
                        {"completed_stages": ["build", "first_call", "quality_pass", "timing", "quality", "manifest"],
                         "note": "the run computed but its artifacts could not be written; no results were published"},
                        run_id=run_doc["run"]["run_id"])
        _print_summary(run_doc, out_dir, written)
        return 0

    finally:
        reservation.release()

if __name__ == "__main__":
    sys.exit(main())
