"""Saved-run artifacts: one authoritative ``run.json`` plus CSV projections of it.

CSV conventions
  * UTF-8, ``\\n`` line endings, minimal quoting (fields containing a comma, quote or newline are quoted, quotes doubled).
  * null  -> blank field.  A blank is NEVER a zero: ``0.0`` is written as ``0.0``.
  * floats are written with ``repr`` (shortest string that round-trips exactly); integers as plain digits.
  * non-finite numbers are rejected (ExportError); booleans are not used.
Units: ``*_ns`` columns are integer nanoseconds measured with ``time.perf_counter_ns``.

``run.json`` is written LAST, so its existence means every other artifact of the run was written successfully.
"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

from .metrics import evaluate_run
from .provenance import sha256_file
from .runner import SUPPORTED_SCHEMA_VERSIONS, median, percentile_nearest_rank

CSV_FILES = ("summary.csv", "per_query.csv", "rankings.csv", "latency.csv", "latency_summary.csv")
RUN_JSON = "run.json"
FAILURE_JSON = "run_failed.json"
FLOAT_TOLERANCE = 1e-12


class ExportError(RuntimeError):
    """Raised when artifacts cannot be written faithfully (non-finite value, I/O failure, ...)."""


def dumps_strict(obj) -> str:
    try:
        return json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    except ValueError as exc:  # json raises ValueError for NaN/Infinity when allow_nan=False
        raise ExportError(f"cannot serialize run to JSON: {exc}") from exc


def _cell(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        raise ExportError("boolean values are not used in CSV artifacts")
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ExportError(f"non-finite value {value!r} cannot be written to CSV")
        return repr(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return value
    raise ExportError(f"unsupported CSV value type {type(value).__name__}")


def csv_tables(run: dict) -> dict[str, tuple[list[str], list[list]]]:
    """Project the run document onto CSV tables: filename -> (header, rows). Pure function of ``run``."""
    methods = run["config"]["methods"]
    ks = run["config"]["ks"]
    query_text = {q["query_id"]: q["text"] for q in run["dataset"]["queries"]}
    summary, per_query, rankings, lat, lat_sum = [], [], [], [], []
    for m in methods:
        res = run["results"][m]
        c = res["counts"]
        for k in ks:
            a = res["aggregate"][str(k)]
            summary.append([m, k, a["macro_recall"], a["mrr"], c["total"], c["scored"], c["no_judged_relevant"]])
        for q in res["queries"]:
            for k in ks:
                met = q["metrics"][str(k)]
                per_query.append([m, q["query_id"], query_text[q["query_id"]], k, q["status"], q["n_relevant"],
                                  q["n_returned"], met["recall"], met["rr"]])
            for r in q["ranking"]:
                rankings.append([m, q["query_id"], r["rank"], r["doc_id"], r["score"], r["label"]])
    for s in run["latency"]["samples"]:
        lat.append([s["method"], s["query_id"], s["repetition"], s["execution_index"], s["position_in_pair"],
                    s["elapsed_ns"]])
    for m in methods:
        o = run["latency"]["summary"][m]["overall"]
        lat_sum.append([m, "overall", None, o["n"], o["median_ns"], o["p95_ns"]])
        for qid, v in run["latency"]["summary"][m]["per_query"].items():
            lat_sum.append([m, "query", qid, v["n"], v["median_ns"], None])  # p95 is only reported overall
    return {
        "summary.csv": (["method", "k", "macro_recall", "mrr", "total_queries", "scored_queries",
                         "no_judged_relevant_queries"], summary),
        "per_query.csv": (["method", "query_id", "query_text", "k", "status", "n_relevant", "n_returned", "recall", "rr"],
                          per_query),
        "rankings.csv": (["method", "query_id", "rank", "doc_id", "score", "label"], rankings),
        "latency.csv": (["method", "query_id", "repetition", "execution_index", "position_in_pair", "elapsed_ns"], lat),
        "latency_summary.csv": (["method", "scope", "query_id", "n_samples", "median_ns", "p95_ns"], lat_sum),
    }


def _write_csv(path: Path, header, rows) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(header)
        for row in rows:
            writer.writerow([_cell(v) for v in row])


def write_run_artifacts(out_dir, run: dict) -> list[Path]:
    """Write run.json + CSVs into ``out_dir`` (must exist and be empty). All-or-nothing: on any failure every file
    written by this call is removed and ExportError is raised. run.json is written last."""
    out_dir = Path(out_dir)
    json_text = dumps_strict(run)  # serialize first: a non-finite value fails before anything touches disk
    tables = csv_tables(run)
    written: list[Path] = []
    tmp_paths: list[Path] = []
    try:
        for name, (header, rows) in tables.items():
            tmp = out_dir / (name + ".tmp")
            tmp_paths.append(tmp)
            _write_csv(tmp, header, rows)
        tmp_json = out_dir / (RUN_JSON + ".tmp")
        tmp_paths.append(tmp_json)
        tmp_json.write_text(json_text, encoding="utf-8", newline="\n")
        for name in CSV_FILES:
            final = out_dir / name
            os.replace(out_dir / (name + ".tmp"), final)
            written.append(final)
        final = out_dir / RUN_JSON
        os.replace(tmp_json, final)  # last: its presence marks a fully written run
        written.append(final)
        return written
    except ExportError:
        _cleanup(written, tmp_paths)
        raise
    except Exception as exc:  # noqa: BLE001
        _cleanup(written, tmp_paths)
        raise ExportError(f"failed to write run artifacts: {type(exc).__name__}: {exc}") from exc


def _cleanup(written, tmp_paths) -> None:
    for p in list(written) + list(tmp_paths):
        try:
            Path(p).unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass


def write_failure_manifest(out_dir, doc: dict) -> Path:
    path = Path(out_dir) / FAILURE_JSON
    path.write_text(dumps_strict(doc), encoding="utf-8", newline="\n")
    return path


# --------------------------------------------------------------------------- reading / verifying
def load_run(path) -> dict:
    def _no_constants(name):
        raise ValueError(f"non-standard JSON constant {name!r} in run file")

    with open(path, encoding="utf-8") as fh:
        return json.load(fh, parse_constant=_no_constants)


def read_csv(path) -> tuple[list[str], list[list[str]]]:
    with open(path, encoding="utf-8", newline="") as fh:
        rows = list(csv.reader(fh))
    return rows[0], rows[1:]


def _close(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= FLOAT_TOLERANCE


def verify_saved_run(run: dict, dataset_dir=None) -> list[str]:
    """Independent consistency checks of a saved run. Returns a list of problems (empty list = consistent)."""
    problems: list[str] = []
    if run.get("schema_version") not in SUPPORTED_SCHEMA_VERSIONS:
        problems.append(f"unsupported schema_version {run.get('schema_version')!r} (supported: {', '.join(SUPPORTED_SCHEMA_VERSIONS)})")
    if run.get("run", {}).get("status") != "completed":
        problems.append("run status is not 'completed'")
    cfg = run["config"]
    methods, ks = cfg["methods"], cfg["ks"]
    ds = run["dataset"]
    query_ids = [q["query_id"] for q in ds["queries"]]
    doc_ids = [d["doc_id"] for d in ds["documents"]]
    qrels: dict[str, dict[str, int]] = {}
    for q, d, r in ds["judgments"]:
        qrels.setdefault(q, {})[d] = r

    expected_samples = len(query_ids) * len(methods) * cfg["measured_per_query_method"]
    samples = run["latency"]["samples"]
    if len(samples) != expected_samples:
        problems.append(f"latency samples: {len(samples)} != expected {expected_samples}")
    if [s["execution_index"] for s in samples] != list(range(1, len(samples) + 1)):
        problems.append("execution_index is not 1..N in order")
    for m in methods:
        for qid in query_ids:
            n = sum(1 for s in samples if s["method"] == m and s["query_id"] == qid)
            if n != cfg["measured_per_query_method"]:
                problems.append(f"{m}/{qid}: {n} measured samples != {cfg['measured_per_query_method']}")

    rc_block = run.get("ranking_consistency")
    if not rc_block or rc_block.get("ranking_disagreements") != 0:
        problems.append(f"ranking_consistency missing or reports ranking disagreements: {rc_block}")
    for m in methods:
        for q in run["results"][m]["queries"]:
            c = q["measured_search_consistency"]
            if c.get("n_ranking_identical") != c.get("n_measured"):
                problems.append(f"{m}/{q['query_id']}: measured searches with a different ordered ID list were saved "
                                "in a completed run")
    counts_seen = set()
    for m in methods:
        res = run["results"][m]
        rankings = {q["query_id"]: [r["doc_id"] for r in q["ranking"]] for q in res["queries"]}
        if list(rankings) != query_ids:
            problems.append(f"{m}: query records {list(rankings)} != dataset queries {query_ids}")
            continue
        recomputed = evaluate_run(query_ids=query_ids, qrels=qrels, rankings=rankings, corpus_ids=doc_ids,
                                  ks=ks)
        rc = (recomputed.total_queries, recomputed.scored_queries, recomputed.no_relevant_queries)
        counts_seen.add(rc)
        sc = res["counts"]
        if (sc["total"], sc["scored"], sc["no_judged_relevant"]) != rc:
            problems.append(f"{m}: saved counts {sc} differ from recomputed {rc}")
        for pq, rec in zip(recomputed.per_query, res["queries"]):
            if rec["n_returned"] != len(rec["ranking"]):
                problems.append(f"{m}/{pq.query_id}: n_returned {rec['n_returned']} != {len(rec['ranking'])} saved results")
            if pq.status != rec["status"]:
                problems.append(f"{m}/{pq.query_id}: status {rec['status']} != recomputed {pq.status}")
            for k in ks:
                saved = rec["metrics"][str(k)]
                if not _close(saved["recall"], pq.recall[k]) or not _close(saved["rr"], pq.rr[k]):
                    problems.append(f"{m}/{pq.query_id}@{k}: saved metrics differ from recomputed")
            for rank, r in enumerate(rec["ranking"], 1):
                if type(r.get("rank")) is not int or r["rank"] != rank:
                    problems.append(f"{m}/{pq.query_id}/{r['doc_id']}: expected rank {rank}, got {r.get('rank')!r}")
                want = {1: "relevant", 0: "nonrelevant", None: "unjudged"}[qrels[pq.query_id].get(r["doc_id"])]
                if r["label"] != want:
                    problems.append(f"{m}/{pq.query_id}/{r['doc_id']}: label {r['label']} != {want}")
        for k in ks:
            a, b = res["aggregate"][str(k)], recomputed.aggregate[k]
            if not _close(a["macro_recall"], b.macro_recall) or not _close(a["mrr"], b.mrr):
                problems.append(f"{m}@{k}: saved aggregate differs from recomputed")
        # latency summaries recomputed from the raw samples
        pooled = [s["elapsed_ns"] for s in samples if s["method"] == m]
        o = run["latency"]["summary"][m]["overall"]
        if o["n"] != len(pooled) or o["median_ns"] != median(pooled) or o["p95_ns"] != percentile_nearest_rank(pooled, 95):
            problems.append(f"{m}: saved overall latency summary differs from recomputed")
        for qid, v in run["latency"]["summary"][m]["per_query"].items():
            vals = [s["elapsed_ns"] for s in samples if s["method"] == m and s["query_id"] == qid]
            if v["n"] != len(vals) or v["median_ns"] != median(vals):
                problems.append(f"{m}/{qid}: saved per-query latency median differs from recomputed")
    if len(counts_seen) > 1:
        problems.append(f"methods disagree on query denominators: {sorted(counts_seen)}")

    if dataset_dir is not None:
        for f in ds.get("files", []):
            p = Path(dataset_dir) / f["name"]
            if not p.is_file() or sha256_file(p) != f["sha256"]:
                problems.append(f"dataset file {f['name']} missing or does not match the saved SHA-256")
    return problems


def compare_runs(a: dict, b: dict) -> dict:
    """Compare two saved runs. Identity/configuration, rankings and quality metrics must match; timestamps, run IDs and
    latencies are expected to differ and are reported, not compared."""
    ident_keys = [
        ("config", lambda r: r["config"]), ("dataset.files", lambda r: r["dataset"].get("files")),
        ("dataset.counts", lambda r: r["dataset"]["counts"]), ("dataset.designation", lambda r: r["dataset"]["designation"]),
        ("adapters", lambda r: r["adapters"]), ("model", lambda r: r["model"]), ("dependencies", lambda r: r["dependencies"]),
        ("code.source_fingerprint", lambda r: r["code"]["source_fingerprint"]["sha256"]),
        ("timing_protocol", lambda r: r["timing_protocol"]),
    ]
    identity_diffs = [name for name, get in ident_keys if get(a) != get(b)]
    ranking_diffs, quality_diffs, max_score_diff = [], [], 0.0
    for m in a["config"]["methods"]:
        for qa, qb in zip(a["results"][m]["queries"], b["results"][m]["queries"]):
            ids_a = [r["doc_id"] for r in qa["ranking"]]
            ids_b = [r["doc_id"] for r in qb["ranking"]]
            if qa["query_id"] != qb["query_id"] or ids_a != ids_b:
                ranking_diffs.append(f"{m}/{qa['query_id']}")
            else:
                for ra, rb in zip(qa["ranking"], qb["ranking"]):
                    max_score_diff = max(max_score_diff, abs(ra["score"] - rb["score"]))
            if qa["metrics"] != qb["metrics"] or qa["status"] != qb["status"]:
                quality_diffs.append(f"{m}/{qa['query_id']}")
        if a["results"][m]["aggregate"] != b["results"][m]["aggregate"] or a["results"][m]["counts"] != b["results"][m]["counts"]:
            quality_diffs.append(f"{m}/aggregate")
    return {
        "identity_equal": not identity_diffs, "identity_differences": identity_diffs,
        "rankings_equal": not ranking_diffs, "ranking_differences": ranking_diffs,
        "max_abs_score_difference": max_score_diff,
        "quality_equal": not quality_diffs, "quality_differences": quality_diffs,
        "run_ids_differ": a["run"]["run_id"] != b["run"]["run_id"],
        "started_differs": a["run"]["started_utc"] != b["run"]["started_utc"],
        "latencies_identical": [s["elapsed_ns"] for s in a["latency"]["samples"]]
        == [s["elapsed_ns"] for s in b["latency"]["samples"]],
    }
