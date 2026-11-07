"""Evaluation runner: one shared query loop over both retrievers, quality metrics, timing protocol.

Design rules (handoff 04):
  * Fail fast and loud: any failure raises ``RunStageError`` carrying stage / method / query. A failed run
    never yields a completed run document; nothing is skipped, substituted, or turned into an empty ranking.
  * Timing wraps exactly ``retriever.search(raw_query_text, max_k)`` and nothing else.
  * The module does not import numpy/torch at import time (thread env vars can be set first by the CLI).
"""

from __future__ import annotations

import math
import time
import traceback
import uuid
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from . import provenance
from .data import Dataset
from .metrics import evaluate_run, validate_ks

SCHEMA_VERSION = "1.1"
# 1.1 changed the setup_observations.embedding cache fields (see README "Setup observation fields"); 1.0 runs stay readable.
SUPPORTED_SCHEMA_VERSIONS = ("1.0", "1.1")
METHODS = ("bm25", "embedding")
DEFAULT_KS = (1, 3, 5)
DEFAULT_WARMUP = 3
DEFAULT_REPETITIONS = 20
DEFAULT_THREADS = 1
MODEL_LICENSE_REFERENCE = {
    "spdx": "Apache-2.0",
    "source": "Hugging Face model tag 'license:apache-2.0', observed 2026-11-07 (not re-queried at run time)",
    "url": "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2",
}
DESIGNATIONS = ("synthetic", "external")

Clock = Callable[[], int]
# A builder takes the clock and returns (retriever, setup_observations). Setup times are measured with it.
Builder = Callable[[Clock], tuple[object, dict]]


class ConfigError(ValueError):
    """Invalid run configuration or input. Raised before any expensive work (CLI exit code 2)."""


class RunStageError(RuntimeError):
    """A run failure with context. Exit code 1; the run is incomplete."""

    def __init__(self, stage: str, method, query_id, original: BaseException):
        self.stage, self.method, self.query_id, self.original = stage, method, query_id, original
        where = ", ".join(f"{k}={v}" for k, v in (("stage", stage), ("method", method), ("query", query_id)) if v)
        super().__init__(f"{where}: {type(original).__name__}: {original}")
        self.partial: dict = {}

    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "method": self.method,
            "query_id": self.query_id,
            "error_type": type(self.original).__name__,
            "message": str(self.original),
            "traceback": "".join(traceback.format_exception(self.original)).strip(),
        }


@dataclass(frozen=True)
class RunConfig:
    dataset_dir: Path
    ks: tuple[int, ...] = DEFAULT_KS
    warmup: int = DEFAULT_WARMUP
    repetitions: int = DEFAULT_REPETITIONS
    threads: int = DEFAULT_THREADS
    designation: str = "synthetic"
    device: str = "cpu"
    methods: tuple[str, ...] = METHODS

    @property
    def max_k(self) -> int:
        return max(self.ks)

    def to_dict(self) -> dict:
        return {
            "dataset_dir": str(self.dataset_dir), "ks": list(self.ks), "max_k": self.max_k,
            "warmup_per_query_method": self.warmup, "measured_per_query_method": self.repetitions,
            "threads_requested": self.threads, "device": self.device, "methods": list(self.methods),
        }


def parse_ks(text: str) -> tuple[int, ...]:
    parts = [p.strip() for p in str(text).split(",")]
    if not parts or any(p == "" for p in parts):
        raise ConfigError(f"--k must be a comma-separated list of positive integers, got {text!r}")
    ks = []
    for p in parts:
        if not p.isascii() or not p.isdigit():
            raise ConfigError(f"--k entries must be positive integers, got {p!r}")
        ks.append(int(p))
    return validate_config_ks(ks)


def validate_config_ks(ks) -> tuple[int, ...]:
    try:
        return validate_ks(ks)
    except ValueError as exc:
        raise ConfigError(f"invalid k values: {exc}") from exc


def validate_count(name: str, value, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{name} must be an integer, got {value!r}")
    if value < minimum:
        raise ConfigError(f"{name} must be >= {minimum}, got {value}")
    return value


def make_config(*, dataset_dir, ks=DEFAULT_KS, warmup=DEFAULT_WARMUP, repetitions=DEFAULT_REPETITIONS,
                threads=DEFAULT_THREADS, designation="synthetic") -> RunConfig:
    if designation not in DESIGNATIONS:
        raise ConfigError(f"dataset designation must be one of {DESIGNATIONS}, got {designation!r}")
    return RunConfig(
        dataset_dir=Path(dataset_dir),
        ks=validate_config_ks(ks),
        warmup=validate_count("warmup", warmup, minimum=0),
        repetitions=validate_count("repetitions", repetitions, minimum=1),
        threads=validate_count("threads", threads, minimum=1),
        designation=designation,
    )


# ---------------------------------------------------------------- statistics
def median(values: Sequence[float]) -> float:
    """Middle value of the sorted sample; for an even count, the mean of the two middle values."""
    if not values:
        raise ValueError("median of an empty sample")
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return float(s[mid]) if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def percentile_nearest_rank(values: Sequence[float], percent: int) -> float:
    """Nearest-rank percentile: the value at 1-based rank ceil(percent/100 * n) of the sorted sample."""
    if not values:
        raise ValueError("percentile of an empty sample")
    if isinstance(percent, bool) or not isinstance(percent, int) or not 0 < percent <= 100:
        raise ValueError(f"percent must be an integer in (0, 100], got {percent!r}")
    s = sorted(values)
    rank = max(1, (percent * len(s) + 99) // 100)  # integer ceil(percent * n / 100)
    return float(s[rank - 1])


def summarize_samples(values: Sequence[int], *, with_p95: bool) -> dict:
    out = {"n": len(values), "median_ns": median(values), "min_ns": min(values), "max_ns": max(values)}
    if with_p95:
        out["p95_ns"] = percentile_nearest_rank(values, 95)
    return out


# ---------------------------------------------------------------- model cache verification
# Byte size of every file selected by retrievers.MODEL_FILE_PATTERNS at the pinned revision, recorded on 2026-11-07 from a
# clean download into an empty cache (identical in a second, independent cache). A snapshot directory that merely exists
# proves nothing: it can be empty or hold a partial download.
PINNED_MODEL_FILES = {
    "1_Pooling/config.json": 190,
    "config.json": 612,
    "config_sentence_transformers.json": 116,
    "data_config.json": 39265,
    "model.safetensors": 90868376,
    "modules.json": 349,
    "sentence_bert_config.json": 53,
    "special_tokens_map.json": 112,
    "tokenizer.json": 466247,
    "tokenizer_config.json": 350,
    "vocab.txt": 231508,
}
CACHE_STATES = ("verified_complete", "partial", "not_cached", "unknown")
CACHE_VERIFICATION_METHOD = ("presence and exact byte size of every file selected by the download patterns, against a table "
                             "recorded for the pinned revision; not a content hash")


def probe_model_cache(resolver: Callable[[], str], *, expected_files: dict | None = None,
                      not_found: tuple = (FileNotFoundError,)) -> dict:
    """Classify the local model cache BEFORE the timed fetch. ``resolver`` returns the local snapshot directory or raises.

    verified_complete  every expected file is present with its expected byte size
    partial            the snapshot directory exists but files are missing or have the wrong size (e.g. an empty folder)
    not_cached         no local snapshot could be resolved (``not_found`` was raised)
    unknown            the state could not be established (unexpected error); treated as "may include a download"
    """
    expected = PINNED_MODEL_FILES if expected_files is None else expected_files
    base = {"method": CACHE_VERIFICATION_METHOD, "files_expected": len(expected)}
    try:
        path = Path(resolver())
    except not_found as exc:
        return {**base, "state": "not_cached", "detail": "no local snapshot directory for the pinned revision"}
    except Exception as exc:  # noqa: BLE001 - an unexpected probe failure is "unknown", never "cached"
        return {**base, "state": "unknown", "detail": f"probe failed: {type(exc).__name__}: {exc}"}
    if not path.is_dir():
        return {**base, "state": "unknown", "detail": f"resolved path is not a directory: {path.name}"}
    missing, mismatched = [], []
    for name, size in expected.items():
        try:
            actual = (path / name).stat().st_size  # follows symlinks; a dangling link raises FileNotFoundError
        except (FileNotFoundError, NotADirectoryError):
            missing.append(name)
            continue
        except OSError as exc:
            return {**base, "state": "unknown", "snapshot_dir_name": path.name,
                    "detail": f"could not inspect {name}: {type(exc).__name__}: {exc}"}
        if actual != size:
            mismatched.append({"file": name, "expected_bytes": size, "actual_bytes": actual})
    state = "verified_complete" if not missing and not mismatched else "partial"
    return {**base, "state": state, "snapshot_dir_name": path.name,
            "files_present_with_expected_size": len(expected) - len(missing) - len(mismatched),
            "missing": missing, "size_mismatch": mismatched}


def pinned_snapshot_dir() -> Path:
    """Where the hub cache keeps the pinned revision (whether or not it exists): <HF_HUB_CACHE>/models--org--name/snapshots/<sha>."""
    from huggingface_hub import constants
    from huggingface_hub.file_download import repo_folder_name

    from .retrievers import DEFAULT_MODEL_ID, PINNED_MODEL_REVISION

    return (Path(constants.HF_HUB_CACHE) / repo_folder_name(repo_id=DEFAULT_MODEL_ID, repo_type="model")
            / "snapshots" / PINNED_MODEL_REVISION)


def probe_pinned_model_cache() -> dict:
    """Classify the real local hub cache for the pinned model WITHOUT asking the hub library for its opinion.

    snapshot_download(local_files_only=True) is not a reliable probe in either direction: it can return an empty or partial
    folder as if it were usable, and (in the locked version) it raises IncompleteSnapshotError for a snapshot that merely lacks
    one file. So the snapshot directory is inspected directly and every selected file is checked against the size table.
    The hub is only consulted for where its cache lives. No network access.
    """
    try:
        snap = pinned_snapshot_dir()
    except Exception as exc:  # noqa: BLE001 - e.g. the hub library layout changed: state unknown, never "cached"
        return {"method": CACHE_VERIFICATION_METHOD, "files_expected": len(PINNED_MODEL_FILES), "state": "unknown",
                "detail": f"could not locate the hub cache: {type(exc).__name__}: {exc}"}

    def resolver() -> str:
        if not snap.is_dir():
            raise FileNotFoundError(snap)
        return str(snap)

    return probe_model_cache(resolver)


def describe_fetch(state: str) -> tuple[bool, str]:
    """(may_include_download, note) for the timed snapshot fetch given the verified pre-run cache state."""
    if state == "verified_complete":
        return False, "all selected files were verified present before the fetch; the fetch was a local resolution check"
    if state == "partial":
        return True, "the snapshot was incomplete before the fetch; the fetch may have downloaded the missing files"
    if state == "not_cached":
        return True, "no local snapshot existed; the fetch includes a download"
    return True, "the cache state could not be verified; the fetch MAY include a download"


# ---------------------------------------------------------------- default (real) builders
def default_builders(dataset: Dataset, config: RunConfig) -> list[tuple[str, Builder]]:
    """Real BM25 + embedding builders. Imports numpy/torch/sentence-transformers lazily, only when called."""

    def build_bm25(clock: Clock):
        from .retrievers import BM25Retriever

        t0 = clock()
        retriever = BM25Retriever(dataset.documents)
        return retriever, {"bm25_index_build_ns": clock() - t0}

    def build_embedding(clock: Clock):
        t_import = clock()  # first-use imports (torch, sentence-transformers, ...) are timed apart from model loading
        import sentence_transformers  # noqa: F401
        import torch
        from huggingface_hub import snapshot_download

        from .retrievers import (DEFAULT_MODEL_ID, MODEL_FILE_PATTERNS, PINNED_MODEL_REVISION,
                                 EmbeddingRetriever, SentenceTransformerEncoder)

        import_ns = clock() - t_import
        torch.set_num_threads(config.threads)
        try:
            torch.set_num_interop_threads(config.threads)
        except RuntimeError:
            pass  # can only be set once, before parallel work has started; observed value is recorded below
        cache = probe_pinned_model_cache()  # verified BEFORE the timed fetch; a resolved folder alone proves nothing
        cache_state = cache["state"]
        may_download, fetch_note = describe_fetch(cache_state)
        t0 = clock()
        snapshot_download(DEFAULT_MODEL_ID, revision=PINNED_MODEL_REVISION, allow_patterns=MODEL_FILE_PATTERNS)
        fetch_ns = clock() - t0
        t1 = clock()
        encoder = SentenceTransformerEncoder(DEFAULT_MODEL_ID, revision=PINNED_MODEL_REVISION, device=config.device)
        load_ns = clock() - t1
        t2 = clock()
        retriever = EmbeddingRetriever(dataset.documents, encoder)
        index_ns = clock() - t2
        return retriever, {
            "library_import_ns": import_ns,
            "model_cache_state_before_run": cache_state,
            "model_cache_verification": cache,
            "model_snapshot_fetch_ns": fetch_ns,
            "model_snapshot_fetch_may_include_download": may_download,
            "model_snapshot_fetch_note": fetch_note,
            "model_load_ns": load_ns,
            "document_embedding_index_build_ns": index_ns,
            "torch_threads_observed": torch.get_num_threads(),
        }

    return [("bm25", build_bm25), ("embedding", build_embedding)]


# ---------------------------------------------------------------- the run
@contextmanager
def _guard(stage, method=None, query_id=None):
    try:
        yield
    except RunStageError:
        raise
    except Exception as exc:  # noqa: BLE001 - re-raised with context
        raise RunStageError(stage, method, query_id, exc) from exc


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _split_result(result, max_k, known_ids):
    """Validate one search result list and return (ids, scores). Anything off is an adapter-contract violation that
    fails the run at once (non-finite scores, unknown/duplicate IDs, too many results); nothing is repaired."""
    if not isinstance(result, list):
        raise TypeError(f"search() must return a list of (doc_id, score), got {type(result).__name__}")
    if len(result) > max_k:
        raise ValueError(f"search() returned {len(result)} results for k={max_k}")
    ids, scores = [], []
    for item in result:
        doc_id, score = item
        if not isinstance(doc_id, str):
            raise TypeError(f"doc_id {doc_id!r} is not a string")
        if doc_id not in known_ids:
            raise ValueError(f"unknown doc_id {doc_id!r} in search result")
        if doc_id in ids:
            raise ValueError(f"duplicate doc_id {doc_id!r} in search result")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError(f"non-finite or non-numeric score {score!r} for {doc_id!r}")
        ids.append(doc_id)
        scores.append(float(score))
    return ids, scores


MAX_DISAGREEMENT_DETAILS = 50
CONSISTENCY_RULE = ("Every first-call, warm-up and measured search must return the same ORDERED document IDs as the untimed "
                    "quality-pass ranking of that (method, query). A different ordered ID list makes the run incomplete "
                    "(stage 'consistency', run_failed.json with the diagnostic, no run.json). If the IDs match but scores "
                    "differ, the run may complete and the differences are reported separately.")


class _Consistency:
    """Compares every search result with the quality-pass reference; collects ranking disagreements as diagnostics."""

    def __init__(self):
        self.calls_checked = 0
        self.ranking_disagreements = 0
        self.score_only_differences = 0
        self.max_abs_score_difference = 0.0
        self.details: list[dict] = []

    def check(self, phase, method, query_id, index, ids, scores, reference):
        """Return 'identical' | 'scores_only' | 'ranking' and the max abs score difference (None for 'ranking')."""
        self.calls_checked += 1
        ref_ids = [d for d, _ in reference]
        if ids != ref_ids:
            self.ranking_disagreements += 1
            if len(self.details) < MAX_DISAGREEMENT_DETAILS:
                self.details.append({"phase": phase, "method": method, "query_id": query_id, "call_index": index,
                                     "expected_ids": ref_ids, "observed_ids": list(ids)})
            return "ranking", None
        diff = max((abs(a - b) for a, (_, b) in zip(scores, reference)), default=0.0)
        if diff > 0.0:
            self.score_only_differences += 1
            self.max_abs_score_difference = max(self.max_abs_score_difference, diff)
            return "scores_only", diff
        return "identical", 0.0

    def summary(self) -> dict:
        return {"rule": CONSISTENCY_RULE, "calls_checked": self.calls_checked,
                "ranking_disagreements": self.ranking_disagreements,
                "score_only_differences": self.score_only_differences,
                "max_abs_score_difference": self.max_abs_score_difference}


def run_evaluation(config: RunConfig, dataset: Dataset, builders: Sequence[tuple[str, Builder]], *,
                   clock: Clock = time.perf_counter_ns, command: Sequence[str] | None = None,
                   dataset_info: dict | None = None) -> dict:
    """Run the full evaluation; return the JSON-ready run document. Raises RunStageError on any failure."""
    methods = [name for name, _ in builders]
    if tuple(methods) != tuple(config.methods):
        raise ConfigError(f"builders {methods} do not match configured methods {list(config.methods)}")
    run_id, started = uuid.uuid4().hex, _utc_now()
    max_k, ks = config.max_k, config.ks
    queries = list(dataset.queries)
    first_query = queries[0]
    known = frozenset(dataset.doc_ids)
    partial = {"completed_stages": [], "setup_observations": {}, "first_call_observations": [],
               "warmup_calls_done": 0, "measured_samples_collected": 0}
    try:
        # ---- build each index once (setup), then record the FIRST retrieval call after each build
        retrievers, setup, first_calls = {}, {}, []
        first_results: dict[str, tuple] = {}
        checker = _Consistency()
        for name, build in builders:
            with _guard("build", name):
                retrievers[name], setup[name] = build(clock)
            partial["setup_observations"][name] = setup[name]
            partial["completed_stages"].append(f"build:{name}")
            with _guard("first_call", name, first_query.query_id):
                t0 = clock()
                result = retrievers[name].search(first_query.text, max_k)
                elapsed = clock() - t0
                first_ids, first_scores = _split_result(result, max_k, known)
            first_results[name] = (first_ids, first_scores)
            first_calls.append({"method": name, "query_id": first_query.query_id, "k": max_k, "elapsed_ns": elapsed,
                                "note": "first retrieval call after this method's index build; not a cold-machine "
                                        "or cold-download measurement"})
            partial["first_call_observations"] = first_calls
            partial["completed_stages"].append(f"first_call:{name}")

        # ---- untimed quality pass: one max-k ranking per (method, query)
        quality_results: dict[str, dict[str, list]] = {m: {} for m in methods}
        rankings: dict[str, dict[str, list[str]]] = {m: {} for m in methods}
        scores_by: dict[str, dict[str, list[float]]] = {m: {} for m in methods}
        for q in queries:
            for m in methods:
                with _guard("retrieve", m, q.query_id):
                    result = retrievers[m].search(q.text, max_k)
                    ids, scores = _split_result(result, max_k, known)
                quality_results[m][q.query_id] = list(zip(ids, scores))
                rankings[m][q.query_id], scores_by[m][q.query_id] = ids, scores
        for m in methods:  # first call (q01) vs the quality-pass ranking of the same (method, query)
            ids, scores = first_results[m]
            checker.check("first_call", m, first_query.query_id, 1, ids, scores, quality_results[m][first_query.query_id])
        partial["completed_stages"].append("quality_pass")

        # ---- timing: per query, warm-up then measured repetitions with rotating method order
        warmup_samples, samples = [], []
        consistency = {m: {q.query_id: {"n_measured": 0, "n_identical_to_quality_pass": 0, "n_ranking_identical": 0,
                                        "n_score_differs_same_ranking": 0, "max_abs_score_difference": 0.0}
                           for q in queries} for m in methods}
        execution_index = 0
        n_methods = len(methods)
        for q in queries:
            for w in range(config.warmup):
                for m in methods:  # warm-up always runs in the listed method order; results are discarded
                    with _guard("warmup", m, q.query_id):
                        t0 = clock()
                        result = retrievers[m].search(q.text, max_k)
                        elapsed = clock() - t0
                        w_ids, w_scores = _split_result(result, max_k, known)
                    checker.check("warmup", m, q.query_id, w + 1, w_ids, w_scores, quality_results[m][q.query_id])
                    warmup_samples.append({"method": m, "query_id": q.query_id, "warmup_index": w + 1,
                                           "elapsed_ns": elapsed})
                    partial["warmup_calls_done"] += 1
            for rep in range(config.repetitions):
                shift = rep % n_methods  # rep 0 starts with methods[0], rep 1 with methods[1], ... (alternates)
                order = methods[shift:] + methods[:shift]
                for pos, m in enumerate(order, 1):
                    execution_index += 1
                    with _guard("measure", m, q.query_id):
                        t0 = clock()                                   # <- timed region start
                        result = retrievers[m].search(q.text, max_k)   # raw query text in, ranked results out
                        elapsed = clock() - t0                         # <- timed region end
                        m_ids, m_scores = _split_result(result, max_k, known)
                    samples.append({"method": m, "query_id": q.query_id, "repetition": rep + 1,
                                    "execution_index": execution_index, "position_in_pair": pos,
                                    "elapsed_ns": elapsed})
                    c = consistency[m][q.query_id]
                    c["n_measured"] += 1
                    c["n_identical_to_quality_pass"] += int(result == quality_results[m][q.query_id])
                    verdict, diff = checker.check("measure", m, q.query_id, rep + 1, m_ids, m_scores,
                                                  quality_results[m][q.query_id])
                    if verdict != "ranking":
                        c["n_ranking_identical"] += 1
                    if verdict == "scores_only":
                        c["n_score_differs_same_ranking"] += 1
                        c["max_abs_score_difference"] = max(c["max_abs_score_difference"], diff)
                    partial["measured_samples_collected"] = len(samples)
        partial["completed_stages"].append("timing")
        if checker.ranking_disagreements:
            first = checker.details[0]
            partial["ranking_disagreements"] = {"total": checker.ranking_disagreements,
                                                "details_kept": len(checker.details),
                                                "details": checker.details, **checker.summary()}
            raise RunStageError(
                "consistency", first["method"], first["query_id"],
                RuntimeError(f"{checker.ranking_disagreements} search call(s) returned a different ordered list of "
                             f"document IDs than the quality-pass ranking (first: {first['phase']} call "
                             f"{first['call_index']}); the run is incomplete and no results are published"))

        # ---- quality metrics: identical query set, qrels, cutoffs and denominators for every method
        metrics = {}
        for m in methods:
            with _guard("quality", m):
                metrics[m] = evaluate_run(query_ids=dataset.query_ids, qrels=dataset.qrels, rankings=rankings[m],
                                          corpus_ids=dataset.doc_ids, ks=ks)
        counts = {m: (metrics[m].total_queries, metrics[m].scored_queries, metrics[m].no_relevant_queries)
                  for m in methods}
        if len(set(counts.values())) != 1:
            raise RunStageError("quality", None, None, RuntimeError(f"methods disagree on denominators: {counts}"))
        partial["completed_stages"].append("quality")

        with _guard("manifest"):
            finished = _utc_now()
            run_doc = _build_run_document(
                config=config, dataset=dataset, dataset_info=dataset_info or {}, retrievers=retrievers,
                setup=setup, first_calls=first_calls, metrics=metrics, rankings=rankings, scores_by=scores_by,
                warmup_samples=warmup_samples, samples=samples, consistency=consistency, consistency_summary=checker.summary(), run_id=run_id,
                started=started, finished=finished, command=list(command or []))
        return run_doc
    except RunStageError as err:
        err.partial = {**partial, "run_id": run_id, "started_utc": started}
        raise


def _build_run_document(*, config, dataset, dataset_info, retrievers, setup, first_calls, metrics, rankings,
                        scores_by, warmup_samples, samples, consistency, consistency_summary, run_id, started, finished, command) -> dict:
    methods = list(config.methods)
    queries = list(dataset.queries)
    n_judgments = sum(len(v) for v in dataset.qrels.values())
    n_positive = sum(1 for v in dataset.qrels.values() for r in v.values() if r == 1)
    adapters = {m: retrievers[m].describe() for m in methods}
    encoder = (adapters.get("embedding") or {}).get("encoder") or {}
    model = None
    if encoder.get("model_id"):
        model = {"model_id": encoder["model_id"], "revision": encoder.get("revision"),
                 "license_reference": MODEL_LICENSE_REFERENCE if "all-MiniLM-L6-v2" in encoder["model_id"]
                 else provenance.UNKNOWN}
    environment = provenance.collect_environment()

    results = {}
    for m in methods:
        run_m = metrics[m]
        q_records = []
        for pq in run_m.per_query:
            qid = pq.query_id
            ids, scs = rankings[m][qid], scores_by[m][qid]
            q_records.append({
                "query_id": qid, "status": pq.status, "n_relevant": pq.n_relevant, "n_returned": pq.n_returned,
                "metrics": {str(k): {"recall": pq.recall[k], "rr": pq.rr[k]} for k in run_m.ks},
                "ranking": [{"rank": i, "doc_id": d, "score": s, "label": dataset.label(qid, d)}
                            for i, (d, s) in enumerate(zip(ids, scs), 1)],
                "measured_search_consistency": consistency[m][qid],
            })
        results[m] = {
            "counts": {"total": run_m.total_queries, "scored": run_m.scored_queries,
                       "no_judged_relevant": run_m.no_relevant_queries},
            "aggregate": {str(k): {"macro_recall": a.macro_recall, "mrr": a.mrr} for k, a in run_m.aggregate.items()},
            "queries": q_records,
        }

    summary = {}
    for m in methods:
        pooled = [s["elapsed_ns"] for s in samples if s["method"] == m]
        per_query = {q.query_id: summarize_samples([s["elapsed_ns"] for s in samples
                                                    if s["method"] == m and s["query_id"] == q.query_id],
                                                   with_p95=False) for q in queries}
        summary[m] = {"overall": summarize_samples(pooled, with_p95=True), "per_query": per_query}

    return {
        "schema_version": SCHEMA_VERSION,
        "run": {"run_id": run_id, "status": "completed", "started_utc": started, "finished_utc": finished,
                "command": command},
        "config": config.to_dict(),
        "dataset": {
            "designation": config.designation,
            "synthetic_developer_fixture": config.designation == "synthetic",
            **dataset_info,
            "counts": {"documents": len(dataset.documents), "queries": len(queries), "judgments": n_judgments,
                       "positive_judgments": n_positive,
                       "queries_without_relevant": sum(1 for q in queries if not dataset.relevant_ids(q.query_id))},
            "documents": [{"doc_id": d.doc_id, "text": d.text} for d in dataset.documents],
            "queries": [{"query_id": q.query_id, "text": q.text} for q in queries],
            "judgments": [[q, d, r] for q, per in dataset.qrels.items() for d, r in per.items()],
        },
        "policies": {
            "metrics": "Recall@k and RR@k per query; macro Recall@k and MRR@k = arithmetic mean over queries with at "
                       "least one judged-relevant document (same query set and denominator for every method).",
            "no_judged_relevant": "metrics are null (JSON null / blank CSV), status 'no_judged_relevant'; excluded from "
                                  "macro means; counted in 'no_judged_relevant'. A null is never a zero.",
            "empty_ranking_for_answerable_query": "scores 0 and stays in the macro denominator.",
            "unjudged_documents": "contribute zero gain, keep their rank, labeled 'unjudged' (distinct from an explicit "
                                  "0 judgment, labeled 'nonrelevant').",
            "ranking_order": "score descending, then doc_id ascending (case-sensitive), applied to the full ranking before "
                             "truncation to max_k. Scores are comparable only within one method.",
            "ranking_consistency": CONSISTENCY_RULE,
            "json": "NaN/Infinity are rejected; absent values are null.",
        },
        "environment": environment,
        "dependencies": provenance.collect_dependencies(),
        "code": provenance.collect_code_identity(),
        "adapters": adapters,
        "model": model,
        "timing_protocol": {
            "clock": "time.perf_counter_ns", "unit": "ns", "k_searched": config.max_k,
            "warmup_per_query_method": config.warmup, "measured_per_query_method": config.repetitions,
            "query_order": [q.query_id for q in queries],
            "method_order_rule": "Per query: warm-up calls first, always in listed method order [%s] and discarded; then "
                                 "measured repetitions. In measured repetition i (0-based) the methods run in the listed "
                                 "order rotated left by (i mod %d), i.e. the first method alternates each repetition."
                                 % (", ".join(methods), len(methods)),
            "timed_region": "raw query text in -> ranked (doc_id, score) list out: query tokenization/encoding, scoring, "
                            "ordering and top-k, as implemented inside retriever.search().",
            "excluded_from_timing": ["data reading", "metric computation", "disk writing", "document text lookup",
                                     "result validation/comparison", "UI"],
            "serial": True, "concurrent_methods": False, "query_cache": False,
            "threads": {"requested": config.threads, "env": environment["thread_env"],
                        "torch_observed": environment["torch_threads_observed"]},
            "median": "middle value of the sorted sample; mean of the two middle values when the count is even",
            "p95": "nearest rank: sorted value at 1-based rank ceil(0.95 * n)",
            "overall_summary": "pools all measured calls of one method (equal repetitions per query)",
            "caveats": "warm, local, serial microbenchmark observations on one machine. Repeated calls of the same query "
                       "are not independent samples and are not evidence of statistical significance. No latency target.",
            "gc": "garbage collection not modified",
        },
        "ranking_consistency": consistency_summary,
        "setup_observations": setup,
        "first_call_observations": first_calls,
        "latency": {"unit": "ns", "samples": samples, "warmup_samples": warmup_samples, "summary": summary},
        "results": results,
    }
