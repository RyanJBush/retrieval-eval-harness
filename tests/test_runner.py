"""Runner tests. Retrievers and the clock are FAKES (see tests/fakes.py); no model, no network, and no assertion
about real latency magnitudes."""

import math
import statistics
from pathlib import Path

import pytest
from fakes import FakeClock, FakeRetriever, default_ranking, make_builders

from retrieval_eval.data import load_dataset
from retrieval_eval.metrics import evaluate_run
from retrieval_eval.runner import (
    ConfigError,
    RunStageError,
    make_config,
    median,
    parse_ks,
    percentile_nearest_rank,
    run_evaluation,
)

FIXTURE = Path(__file__).resolve().parents[1] / "data" / "synthetic"


@pytest.fixture(scope="module")
def dataset():
    return load_dataset(FIXTURE)


def go(dataset, *, warmup=3, repetitions=20, ks=(1, 3, 5), per_method=None, events=None, clock=None):
    clock = clock or FakeClock(events=events)
    cfg = make_config(dataset_dir=FIXTURE, ks=ks, warmup=warmup, repetitions=repetitions)
    builders = make_builders(clock, dataset.doc_ids, events=events, per_method=per_method)
    return run_evaluation(cfg, dataset, builders, clock=clock), clock


# ------------------------------------------------------------------ statistics (independent expectations)
@pytest.mark.parametrize("values", [[5], [1, 2], [3, 1, 2], [4, 1, 3, 2], [7, 7, 7, 7], [10, 20], list(range(20))])
def test_median_matches_stdlib(values):
    assert median(values) == statistics.median(values)


def test_median_even_count_is_mean_of_middle_two_and_rejects_empty():
    assert median([1, 2, 3, 10]) == 2.5
    with pytest.raises(ValueError):
        median([])


@pytest.mark.parametrize("n", [1, 2, 10, 19, 20, 21, 100, 160])
def test_p95_nearest_rank(n):
    values = list(range(1, n + 1))
    rank = math.ceil(0.95 * n)  # independent: ceil(0.95 * n), 1-based
    assert percentile_nearest_rank(values[::-1], 95) == float(values[rank - 1])


def test_p95_known_values():
    assert percentile_nearest_rank(list(range(1, 21)), 95) == 19.0  # ceil(19.0) = 19
    assert percentile_nearest_rank(list(range(1, 161)), 95) == 152.0  # ceil(152.0) = 152
    assert percentile_nearest_rank([42], 95) == 42.0
    for bad in (0, 101, -5, 95.0, True):
        with pytest.raises(ValueError):
            percentile_nearest_rank([1, 2], bad)
    with pytest.raises(ValueError):
        percentile_nearest_rank([], 95)


# ------------------------------------------------------------------ configuration
def test_parse_ks_and_config_validation():
    assert parse_ks("1,3,5") == (1, 3, 5) and parse_ks(" 2 , 4 ") == (2, 4)
    for bad in ("", "1,,3", "0", "-1", "1,1", "a", "1.5", "1;2", "٣"):
        with pytest.raises(ConfigError):
            parse_ks(bad)
    for kw in ({"repetitions": 0}, {"repetitions": -3}, {"repetitions": 1.5}, {"repetitions": True},
               {"warmup": -1}, {"warmup": 2.0}, {"threads": 0}, {"designation": "other"}):
        with pytest.raises(ConfigError):
            make_config(dataset_dir=FIXTURE, **kw)
    assert make_config(dataset_dir=FIXTURE, warmup=0).warmup == 0  # zero warm-ups is allowed
    cfg = make_config(dataset_dir=FIXTURE, ks=(5, 1, 3))
    assert cfg.max_k == 5 and cfg.ks == (5, 1, 3)


# ------------------------------------------------------------------ sample counts and schedule
def test_default_protocol_sample_counts(dataset):
    run, _ = go(dataset)
    samples = run["latency"]["samples"]
    assert len(samples) == 320 == 8 * 2 * 20
    assert len(run["latency"]["warmup_samples"]) == 48 == 8 * 2 * 3  # kept separate, not in the 320
    assert len(run["first_call_observations"]) == 2
    for m in ("bm25", "embedding"):
        for q in dataset.query_ids:
            assert sum(1 for s in samples if s["method"] == m and s["query_id"] == q) == 20
    assert {s["query_id"] for s in samples} == set(dataset.query_ids)  # includes the no-relevant query (q08)


def test_schedule_order_alternates_methods_and_numbers_executions(dataset):
    run, _ = go(dataset)
    samples = run["latency"]["samples"]
    assert [s["execution_index"] for s in samples] == list(range(1, 321))
    assert [s["query_id"] for s in samples] == [q for q in dataset.query_ids for _ in range(40)]  # query order
    for i in range(0, len(samples), 2):
        a, b = samples[i], samples[i + 1]
        assert a["query_id"] == b["query_id"] and a["repetition"] == b["repetition"]
        assert (a["position_in_pair"], b["position_in_pair"]) == (1, 2)
        rep = a["repetition"]  # 1-based; odd repetitions start with bm25, even with embedding
        first, second = ("bm25", "embedding") if rep % 2 == 1 else ("embedding", "bm25")
        assert (a["method"], b["method"]) == (first, second)
    for m in ("bm25", "embedding"):  # each method leads exactly half of its repetitions
        firsts = [s for s in samples if s["method"] == m and s["position_in_pair"] == 1]
        assert len(firsts) == 80


def test_warmups_run_before_measurements_per_query_in_listed_order(dataset):
    events = []
    run, _ = go(dataset, warmup=2, repetitions=2, events=events)
    # after both first-call + quality-pass phases, the timed calls for q01 are: 2 warm-ups x (bm25, embedding) then
    # 2 measured repetitions that alternate. Check the method sequence of q01's timed calls.
    q01 = dataset.queries[0].text
    timed = [e for i, e in enumerate(events)
             if e[0] == "search" and 0 < i < len(events) - 1 and events[i - 1] == ("clock",)
             and events[i + 1] == ("clock",) and e[2] == q01]
    methods = [e[1] for e in timed][2:]  # drop the two first-call observations (also timed, also q01)
    assert methods == ["bm25", "embedding", "bm25", "embedding",  # warm-ups: fixed listed order
                       "bm25", "embedding", "embedding", "bm25"]  # measured rep 1, then rep 2 flipped


# ------------------------------------------------------------------ timing boundaries with a controlled clock
def test_clock_is_read_exactly_twice_around_each_timed_search_and_nowhere_else(dataset):
    events = []
    run, clock = go(dataset, events=events)
    # fake builders read the clock twice each (4); first calls (2 x 2); warm-ups (48 x 2); measured (320 x 2)
    assert clock.reads == 4 + 2 * (2 + 48 + 320)
    searches = [i for i, e in enumerate(events) if e[0] == "search"]
    is_timed = lambda i: events[i - 1] == ("clock",) and events[i + 1] == ("clock",)  # noqa: E731
    timed = [i for i in searches if is_timed(i)]
    untimed = [i for i in searches if not is_timed(i)]
    for i in timed:
        assert events[i + 1] == ("clock",)  # clock, search, clock with nothing in between
    assert len(timed) == 2 + 48 + 320
    assert len(untimed) == 16  # the quality pass: 8 queries x 2 methods, no timing around them
    assert all(events[i][3] == 5 for i in searches)  # always searched at max k
    texts = {q.text for q in dataset.queries}
    assert all(events[i][2] in texts for i in searches)  # raw query text, exactly as stored


def test_measured_durations_equal_scripted_costs_and_exclude_untimed_work(dataset):
    warmup, reps = 3, 20
    run, _ = go(dataset, warmup=warmup, repetitions=reps)
    base = FakeRetriever.BASE
    for m in ("bm25", "embedding"):
        for qi, q in enumerate(dataset.query_ids):
            # prior calls of (m, q): first-call (only q01) + quality pass + warm-ups; measured rep r is next
            prior = (1 if qi == 0 else 0) + 1 + warmup
            got = [s["elapsed_ns"] for s in run["latency"]["samples"] if s["method"] == m and s["query_id"] == q]
            assert got == [base[m] + 10 * (prior + r + 1) for r in range(reps)]
    # describe() advances the fake clock by 1e9 outside the timed region; it must not appear anywhere
    assert max(s["elapsed_ns"] for s in run["latency"]["samples"]) < 10**6


def test_latency_summaries_match_independent_computation(dataset):
    run, _ = go(dataset)
    for m in ("bm25", "embedding"):
        pooled = [s["elapsed_ns"] for s in run["latency"]["samples"] if s["method"] == m]
        o = run["latency"]["summary"][m]["overall"]
        assert o["n"] == 160 and o["median_ns"] == statistics.median(pooled)
        assert o["p95_ns"] == sorted(pooled)[math.ceil(0.95 * 160) - 1]
        for q in dataset.query_ids:
            vals = [s["elapsed_ns"] for s in run["latency"]["samples"] if s["method"] == m and s["query_id"] == q]
            assert run["latency"]["summary"][m]["per_query"][q]["median_ns"] == statistics.median(vals)  # even count


def test_first_call_and_setup_are_recorded_separately_from_samples(dataset):
    run, _ = go(dataset)
    first = {f["method"]: f for f in run["first_call_observations"]}
    assert set(first) == {"bm25", "embedding"}
    assert all(f["query_id"] == dataset.queries[0].query_id and f["k"] == 5 for f in first.values())
    assert first["bm25"]["elapsed_ns"] == FakeRetriever.BASE["bm25"] + 10 * 1  # the very first call of that retriever
    assert "not a cold-machine" in first["bm25"]["note"]
    assert set(run["setup_observations"]) == {"bm25", "embedding"}
    assert len(run["latency"]["samples"]) == 320  # first calls/warm-ups are not in the measured rows


# ------------------------------------------------------------------ quality
def test_counts_and_metrics_match_evaluate_run(dataset):
    run, _ = go(dataset)
    for m in ("bm25", "embedding"):
        res = run["results"][m]
        assert res["counts"] == {"total": 8, "scored": 7, "no_judged_relevant": 1}
        rankings = {q["query_id"]: [r["doc_id"] for r in q["ranking"]] for q in res["queries"]}
        expected = evaluate_run(query_ids=dataset.query_ids, qrels=dataset.qrels, rankings=rankings,
                                corpus_ids=dataset.doc_ids, ks=(1, 3, 5))
        for k in (1, 3, 5):
            assert res["aggregate"][str(k)]["macro_recall"] == expected.aggregate[k].macro_recall
            assert res["aggregate"][str(k)]["mrr"] == expected.aggregate[k].mrr
        q08 = [q for q in res["queries"] if q["query_id"] == "q08"][0]
        assert q08["status"] == "no_judged_relevant"
        assert q08["metrics"]["3"] == {"recall": None, "rr": None}  # null, not zero
        assert all(len(q["ranking"]) == 5 for q in res["queries"])  # max-k ranking stored
        assert all(r["label"] in {"relevant", "nonrelevant", "unjudged"} for q in res["queries"] for r in q["ranking"])


def test_hand_checked_quality_for_scripted_rankings(dataset):
    # q01 relevant = {d01}; q03 relevant = {d02, d03}; q08 has none. Script bm25 only.
    scripted = {
        dataset.queries[0].text: [("d05", 3.0), ("d01", 2.0), ("d07", 1.0)],          # q01: rank-2 hit
        dataset.queries[2].text: [("d02", 3.0)],                                       # q03: 1 of 2 found
        dataset.queries[7].text: [("d01", 1.0)],                                       # q08: no relevant
    }
    run, _ = go(dataset, per_method={"bm25": {"rankings": scripted}})
    q = {x["query_id"]: x for x in run["results"]["bm25"]["queries"]}
    assert q["q01"]["metrics"]["1"] == {"recall": 0.0, "rr": 0.0}
    assert q["q01"]["metrics"]["3"] == {"recall": 1.0, "rr": 0.5}
    assert q["q03"]["metrics"]["5"] == {"recall": 0.5, "rr": 1.0} and q["q03"]["n_returned"] == 1  # short list, no padding
    assert q["q08"]["metrics"]["5"] == {"recall": None, "rr": None}


def test_empty_ranking_for_answerable_query_scores_zero_and_stays_in_denominator(dataset):
    scripted = {dataset.queries[0].text: []}  # q01 is answerable (relevant d01)
    run, _ = go(dataset, per_method={"embedding": {"rankings": scripted}})
    res = run["results"]["embedding"]
    q01 = [x for x in res["queries"] if x["query_id"] == "q01"][0]
    assert q01["status"] == "scored" and q01["n_returned"] == 0
    assert q01["metrics"]["5"] == {"recall": 0.0, "rr": 0.0}  # zero, explicitly not null
    assert res["counts"]["scored"] == 7  # still counted
    assert run["results"]["bm25"]["counts"] == res["counts"]  # identical denominators across methods


def test_unjudged_and_nonrelevant_labels_are_distinct(dataset):
    # the fixture has complete judgments, so every label is relevant/nonrelevant; labels come from the dataset
    run, _ = go(dataset)
    labels = {r["label"] for m in ("bm25", "embedding") for q in run["results"][m]["queries"] for r in q["ranking"]}
    assert labels <= {"relevant", "nonrelevant"} and "relevant" in labels


def test_identical_repeat_searches_are_recorded_as_consistent(dataset):
    run, _ = go(dataset, warmup=1, repetitions=4)
    for m in ("bm25", "embedding"):
        for q in run["results"][m]["queries"]:
            assert q["measured_search_consistency"] == {
                "n_measured": 4, "n_identical_to_quality_pass": 4, "n_ranking_identical": 4,
                "n_score_differs_same_ranking": 0, "max_abs_score_difference": 0.0}
    rc = run["ranking_consistency"]
    # per method: 1 first call + 8 queries x (1 warm-up + 4 measured) = 41 checked calls; 2 methods
    assert rc["calls_checked"] == 2 * (1 + 8 * 5)
    assert (rc["ranking_disagreements"], rc["score_only_differences"], rc["max_abs_score_difference"]) == (0, 0, 0.0)
    assert "ORDERED document IDs" in run["policies"]["ranking_consistency"]


@pytest.mark.parametrize("phase,kwargs,query_idx", [
    ("measure", {"drift_after": 5}, None),             # later calls reverse the order
    ("warmup", {"reverse_calls": {3}}, 0),             # q01: 1 first call, 2 quality pass, 3 = first warm-up
    ("first_call", {"reverse_calls": {1}}, 0),         # the first retrieval call after the build
])
def test_changed_ordered_ids_make_the_run_incomplete_and_keep_the_diagnostic(dataset, phase, kwargs, query_idx):
    with pytest.raises(RunStageError) as exc:
        go(dataset, warmup=1, repetitions=4, per_method={"embedding": kwargs})
    e = exc.value
    assert e.stage == "consistency" and e.method == "embedding"
    assert "different ordered list of document IDs" in str(e) and "no results are published" in str(e)
    diag = e.partial["ranking_disagreements"]
    assert diag["total"] >= 1 and diag["details"][0]["phase"] == phase and diag["details"][0]["method"] == "embedding"
    d0 = diag["details"][0]
    assert d0["expected_ids"] != d0["observed_ids"] and len(d0["expected_ids"]) == len(d0["observed_ids"]) == 5
    assert "timing" in e.partial["completed_stages"] and "quality" not in e.partial["completed_stages"]
    assert not any(k in e.partial for k in ("results", "metrics"))  # no results are part of the partial record


def test_disagreement_details_are_capped_but_total_is_exact(dataset):
    with pytest.raises(RunStageError) as exc:
        go(dataset, warmup=0, repetitions=20, per_method={"embedding": {"drift_after": 2}})
    diag = exc.value.partial["ranking_disagreements"]
    assert diag["total"] > 50 and diag["details_kept"] == 50 == len(diag["details"])


def test_same_ranking_with_different_scores_completes_and_is_reported_separately(dataset):
    run, _ = go(dataset, warmup=1, repetitions=4, per_method={"bm25": {"jitter_calls": {5}}})
    rc = run["ranking_consistency"]
    assert rc["ranking_disagreements"] == 0 and rc["score_only_differences"] == 8  # call 5 of each of the 8 queries is jittered
    assert 0.0 < rc["max_abs_score_difference"] <= 1e-8
    cells = [q["measured_search_consistency"] for q in run["results"]["bm25"]["queries"]]
    assert all(c["n_ranking_identical"] == c["n_measured"] == 4 for c in cells)
    assert sum(c["n_score_differs_same_ranking"] for c in cells) == 8
    assert all(c["n_identical_to_quality_pass"] == c["n_measured"] - c["n_score_differs_same_ranking"] for c in cells)
    assert run["run"]["status"] == "completed"
    assert all(c["n_score_differs_same_ranking"] == 0 for c in
               [q["measured_search_consistency"] for q in run["results"]["embedding"]["queries"]])


def test_manifest_contents(dataset):
    run, _ = go(dataset)
    assert run["schema_version"] == "1.1" and run["run"]["status"] == "completed" and run["run"]["run_id"]
    assert run["config"]["ks"] == [1, 3, 5] and run["config"]["max_k"] == 5
    assert run["dataset"]["synthetic_developer_fixture"] is True
    assert run["dataset"]["counts"] == {"documents": 20, "queries": 8, "judgments": 160, "positive_judgments": 9,
                                        "queries_without_relevant": 1}
    assert len(run["dataset"]["documents"]) == 20 and len(run["dataset"]["judgments"]) == 160
    assert run["timing_protocol"]["serial"] is True and run["timing_protocol"]["query_cache"] is False
    assert "ceil(0.95" in run["timing_protocol"]["p95"]
    assert run["model"]["model_id"] == "FAKE-encoder"
    assert "numpy" in run["dependencies"] and run["environment"]["python_version"]
    assert run["code"]["source_fingerprint"]["sha256"]


# ------------------------------------------------------------------ failures: stage/method/query context, no silent repair
def boom(msg):
    return RuntimeError(msg)


def test_build_failure_carries_stage_and_method(dataset):
    clock = FakeClock()
    builders = make_builders(clock, dataset.doc_ids)
    def bad(clk):
        raise ConnectionError("model download failed")
    builders[1] = ("embedding", bad)
    cfg = make_config(dataset_dir=FIXTURE)
    with pytest.raises(RunStageError) as exc:
        run_evaluation(cfg, dataset, builders, clock=clock)
    e = exc.value
    assert (e.stage, e.method, e.query_id) == ("build", "embedding", None)
    assert isinstance(e.original, ConnectionError) and "model download failed" in str(e)
    assert e.partial["completed_stages"] == ["build:bm25", "first_call:bm25"]  # progress is reported, not results
    assert "aggregate" not in str(e.partial)


@pytest.mark.parametrize(
    "call_number,stage,query_idx",
    [(1, "first_call", 0),     # the very first search of the retriever
     (2, "retrieve", 0)],      # second search of q01 = the untimed quality pass
)
def test_early_search_failures_are_stage_tagged(dataset, call_number, stage, query_idx):
    q = dataset.queries[query_idx]
    run_kw = {"embedding": {"fail": {(q.text, call_number): boom("embedding exploded")}}}
    with pytest.raises(RunStageError) as exc:
        go(dataset, per_method=run_kw)
    e = exc.value
    assert (e.stage, e.method, e.query_id) == (stage, "embedding", q.query_id)


def test_warmup_failure_is_stage_tagged(dataset):
    q = dataset.queries[3]  # q04: calls = quality pass (1) then warm-ups start at call 2
    with pytest.raises(RunStageError) as exc:
        go(dataset, per_method={"bm25": {"fail": {(q.text, 2): boom("warm-up failure")}}})
    assert (exc.value.stage, exc.value.method, exc.value.query_id) == ("warmup", "bm25", "q04")


def test_measurement_failure_is_stage_tagged_and_run_is_not_completed(dataset):
    q = dataset.queries[5]  # q06: call 1 quality pass, 2-4 warm-ups, 5.. measured
    with pytest.raises(RunStageError) as exc:
        go(dataset, per_method={"embedding": {"fail": {(q.text, 9): boom("mid-run failure")}}})
    e = exc.value
    assert (e.stage, e.method, e.query_id) == ("measure", "embedding", "q06")
    assert e.partial["measured_samples_collected"] > 0 and "timing" not in e.partial["completed_stages"]
    assert e.to_dict()["error_type"] == "RuntimeError" and "mid-run failure" in e.to_dict()["traceback"]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), "x", None])
def test_nonfinite_or_bad_scores_are_a_stage_failure_not_a_result(dataset, bad):
    scripted = {dataset.queries[1].text: [("d10", bad)]}
    with pytest.raises(RunStageError) as exc:
        go(dataset, per_method={"bm25": {"rankings": scripted}})
    assert (exc.value.stage, exc.value.method, exc.value.query_id) == ("retrieve", "bm25", "q02")


def test_unknown_or_duplicate_doc_id_fails_fast_not_silently(dataset):
    for bad_rank in ([("zzz", 1.0)], [("d01", 2.0), ("d01", 1.0)]):
        scripted = {dataset.queries[0].text: bad_rank}
        with pytest.raises(RunStageError) as exc:
            go(dataset, per_method={"embedding": {"rankings": scripted}})
        assert (exc.value.stage, exc.value.method, exc.value.query_id) == ("first_call", "embedding", "q01")
        assert isinstance(exc.value.original, ValueError)


def test_quality_stage_failure_is_stage_tagged(dataset, monkeypatch):
    import retrieval_eval.runner as runner_mod

    def explode(**kwargs):
        raise RuntimeError("metric failure")

    monkeypatch.setattr(runner_mod, "evaluate_run", explode)
    with pytest.raises(RunStageError) as exc:
        go(dataset)
    assert (exc.value.stage, exc.value.method) == ("quality", "bm25")
    assert "timing" in exc.value.partial["completed_stages"]


def test_more_results_than_k_are_rejected(dataset):
    class TooMany(FakeRetriever):
        def search(self, query_text, k):
            return default_ranking(self.name, self._doc_ids)[: k + 1]

    clock = FakeClock()
    builders = [("bm25", lambda clk: (TooMany("bm25", dataset.doc_ids, clock), {})),
                ("embedding", lambda clk: (FakeRetriever("embedding", dataset.doc_ids, clock), {}))]
    with pytest.raises(RunStageError) as exc:
        run_evaluation(make_config(dataset_dir=FIXTURE), dataset, builders, clock=clock)
    assert exc.value.stage == "first_call"


def test_builders_must_match_configured_methods(dataset):
    clock = FakeClock()
    with pytest.raises(ConfigError):
        run_evaluation(make_config(dataset_dir=FIXTURE), dataset, make_builders(clock, dataset.doc_ids)[:1], clock=clock)


def test_validation_and_metric_work_after_the_search_is_never_inside_the_timed_region(dataset, monkeypatch):
    import retrieval_eval.runner as runner_mod

    clock = FakeClock()
    real_split, real_eval = runner_mod._split_result, runner_mod.evaluate_run

    def slow_split(*a, **k):
        clock.advance(10**9)  # one second of "validation" after every search
        return real_split(*a, **k)

    def slow_eval(**k):
        clock.advance(10**9)
        return real_eval(**k)

    monkeypatch.setattr(runner_mod, "_split_result", slow_split)
    monkeypatch.setattr(runner_mod, "evaluate_run", slow_eval)
    cfg = make_config(dataset_dir=FIXTURE, warmup=1, repetitions=3)
    run = run_evaluation(cfg, dataset, make_builders(clock, dataset.doc_ids), clock=clock)
    assert max(s["elapsed_ns"] for s in run["latency"]["samples"]) < 10**6  # none of the 1e9 ns leaked in
    assert max(w["elapsed_ns"] for w in run["latency"]["warmup_samples"]) < 10**6
    assert max(f["elapsed_ns"] for f in run["first_call_observations"]) < 10**6


# ------------------------------------------------------------------ model cache verification (setup observation fields)
import fnmatch  # noqa: E402
import sys  # noqa: E402
import types  # noqa: E402

from retrieval_eval import runner as runner_mod  # noqa: E402
from retrieval_eval.runner import PINNED_MODEL_FILES, describe_fetch, probe_model_cache  # noqa: E402

SMALL = {"config.json": 10, "sub/vocab.txt": 20, "weights.bin": 1000}


def make_snapshot(root: Path, files: dict, *, sparse=True):
    root.mkdir(parents=True, exist_ok=True)
    for name, size in files.items():
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as fh:
            fh.truncate(size)  # sparse file of exactly `size` bytes
    return root


def probe(path, **kw):
    return probe_model_cache(lambda: str(path), **kw)


def test_empty_snapshot_directory_is_partial_not_cached(tmp_path):
    """The reproduced case: a resolved folder with no files must never count as a complete cache."""
    snap = tmp_path / "snap"
    snap.mkdir()
    res = probe(snap, expected_files=SMALL)
    assert res["state"] == "partial" and res["files_present_with_expected_size"] == 0
    assert sorted(res["missing"]) == sorted(SMALL) and res["size_mismatch"] == []
    assert describe_fetch(res["state"])[0] is True  # the fetch may include a download


def test_partially_populated_and_wrong_size_snapshots_are_partial(tmp_path):
    snap = make_snapshot(tmp_path / "snap", {"config.json": 10, "weights.bin": 1000})  # one file missing
    res = probe(snap, expected_files=SMALL)
    assert res["state"] == "partial" and res["missing"] == ["sub/vocab.txt"] and res["files_present_with_expected_size"] == 2
    snap2 = make_snapshot(tmp_path / "snap2", {**SMALL, "weights.bin": 999})  # interrupted download: wrong size
    res2 = probe(snap2, expected_files=SMALL)
    assert res2["state"] == "partial" and res2["missing"] == []
    assert res2["size_mismatch"] == [{"file": "weights.bin", "expected_bytes": 1000, "actual_bytes": 999}]


def test_dangling_symlink_counts_as_missing(tmp_path):
    snap = make_snapshot(tmp_path / "snap", {"config.json": 10, "weights.bin": 1000})
    (snap / "sub").mkdir()
    (snap / "sub" / "vocab.txt").symlink_to(tmp_path / "blob-that-does-not-exist")
    res = probe(snap, expected_files=SMALL)
    assert res["state"] == "partial" and res["missing"] == ["sub/vocab.txt"]


def test_exactly_matching_snapshot_is_verified_complete(tmp_path):
    res = probe(make_snapshot(tmp_path / "snap", SMALL), expected_files=SMALL)
    assert res["state"] == "verified_complete" and res["missing"] == [] and res["size_mismatch"] == []
    assert res["files_present_with_expected_size"] == res["files_expected"] == 3
    may, note = describe_fetch(res["state"])
    assert may is False and "verified" in note
    assert "not a content hash" in res["method"]  # the limit of the verification is stated


def test_pinned_table_is_verified_complete_only_when_every_file_has_its_exact_size(tmp_path):
    snap = make_snapshot(tmp_path / "snap", PINNED_MODEL_FILES)
    assert probe(snap)["state"] == "verified_complete"
    (snap / "model.safetensors").unlink()
    assert probe(snap)["state"] == "partial" and probe(snap)["missing"] == ["model.safetensors"]
    with open(snap / "model.safetensors", "wb") as fh:
        fh.truncate(PINNED_MODEL_FILES["model.safetensors"] - 1)
    assert probe(snap)["state"] == "partial" and probe(snap)["size_mismatch"][0]["file"] == "model.safetensors"


def test_pinned_table_matches_the_download_patterns():
    from retrieval_eval.retrievers import MODEL_FILE_PATTERNS

    assert all(isinstance(v, int) and v > 0 for v in PINNED_MODEL_FILES.values())
    for name in PINNED_MODEL_FILES:  # every table entry is something the patterns select
        assert any(fnmatch.fnmatch(name, pat) for pat in MODEL_FILE_PATTERNS), name
    assert "model.safetensors" in PINNED_MODEL_FILES and len(PINNED_MODEL_FILES) == 11


class LocalMiss(FileNotFoundError):
    pass


def test_unresolvable_and_unverifiable_states(tmp_path):
    def miss():
        raise LocalMiss("nothing cached")

    assert probe_model_cache(miss, expected_files=SMALL, not_found=(LocalMiss,))["state"] == "not_cached"
    assert probe_model_cache(miss, expected_files=SMALL)["state"] == "not_cached"  # FileNotFoundError subclass

    def broken():
        raise RuntimeError("hub exploded")

    res = probe_model_cache(broken, expected_files=SMALL)
    assert res["state"] == "unknown" and "hub exploded" in res["detail"]  # an unexpected failure is NOT "not cached"
    assert describe_fetch("unknown")[0] is True and "MAY include" in describe_fetch("unknown")[1]
    afile = tmp_path / "afile"
    afile.write_text("x", encoding="utf-8")
    assert probe(afile, expected_files=SMALL)["state"] == "unknown"  # resolved to something that is not a directory


def test_unreadable_file_makes_the_state_unknown_not_complete(tmp_path, monkeypatch):
    snap = make_snapshot(tmp_path / "snap", SMALL)
    real_stat = Path.stat

    def deny(self, *a, **k):
        if self.name == "weights.bin":
            raise PermissionError("denied")
        return real_stat(self, *a, **k)

    monkeypatch.setattr(Path, "stat", deny)
    res = probe(snap, expected_files=SMALL)
    assert res["state"] == "unknown" and "weights.bin" in res["detail"]


def test_fetch_may_include_download_only_when_completeness_is_verified_false():
    assert [describe_fetch(s)[0] for s in runner_mod.CACHE_STATES] == [False, True, True, True]


def test_embedding_builder_records_verified_state_and_labels_the_fetch(dataset, monkeypatch):
    """Field plumbing of the real builder with stubbed heavy modules (no torch/model needed)."""
    import retrieval_eval.retrievers as retrievers_mod

    calls = []
    fake_torch = types.SimpleNamespace(set_num_threads=lambda n: None, set_num_interop_threads=lambda n: None,
                                       get_num_threads=lambda: 1)
    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.snapshot_download = lambda *a, **k: calls.append(("fetch", k)) or "/unused"
    fake_hub_errors = types.ModuleType("huggingface_hub.errors")
    fake_hub_errors.LocalEntryNotFoundError = LocalMiss
    for name, mod in (("torch", fake_torch), ("sentence_transformers", types.ModuleType("sentence_transformers")),
                      ("huggingface_hub", fake_hub), ("huggingface_hub.errors", fake_hub_errors)):
        monkeypatch.setitem(sys.modules, name, mod)

    class StubEncoder:
        def __init__(self, *a, **k):
            calls.append("load")

    class StubRetriever:
        def __init__(self, docs, enc):
            calls.append("index")

    monkeypatch.setattr(retrievers_mod, "SentenceTransformerEncoder", StubEncoder)
    monkeypatch.setattr(retrievers_mod, "EmbeddingRetriever", StubRetriever)
    cfg = make_config(dataset_dir=FIXTURE)
    expect = {"partial": True, "verified_complete": False, "not_cached": True, "unknown": True}
    for state, may in expect.items():
        calls.clear()
        verification = {"state": state, "files_expected": 11, "method": "m", "missing": ["x"] if state == "partial" else []}
        monkeypatch.setattr(runner_mod, "probe_pinned_model_cache", lambda v=verification: v)
        _, build = runner_mod.default_builders(dataset, cfg)[1]
        retriever, setup = build(FakeClock())
        assert setup["model_cache_state_before_run"] == state and setup["model_cache_verification"] == verification
        assert setup["model_snapshot_fetch_may_include_download"] is may
        assert "model_snapshot_fetch_includes_download" not in setup  # the old, over-confident field is gone
        assert setup["model_snapshot_fetch_note"] and all(k in setup for k in (
            "library_import_ns", "model_snapshot_fetch_ns", "model_load_ns", "document_embedding_index_build_ns"))
        assert [c for c in calls if isinstance(c, tuple)][0][0] == "fetch" and calls[-2:] == ["load", "index"]  # fetch, then load, then index


def test_schema_1_1_is_current_and_1_0_runs_remain_readable(dataset):
    from retrieval_eval import export
    from retrieval_eval.runner import SCHEMA_VERSION, SUPPORTED_SCHEMA_VERSIONS

    run, _ = go(dataset)
    assert SCHEMA_VERSION == "1.1" and run["schema_version"] == "1.1" and SUPPORTED_SCHEMA_VERSIONS == ("1.0", "1.1")
    assert export.verify_saved_run(run) == []
    old = dict(run, schema_version="1.0")  # a 1.0 run differs only in the setup_observations cache fields
    assert export.verify_saved_run(old) == []
    assert any("unsupported schema_version" in p for p in export.verify_saved_run(dict(run, schema_version="2.0")))


def _stub_hub(monkeypatch, cache_root: Path, *, snapshot_download=None):
    """Stub only the two hub names the probe uses: where the cache lives and how a repo folder is named."""
    constants = types.ModuleType("huggingface_hub.constants")
    constants.HF_HUB_CACHE = str(cache_root)
    file_download = types.ModuleType("huggingface_hub.file_download")
    file_download.repo_folder_name = lambda *, repo_id, repo_type: f"{repo_type}s--" + repo_id.replace("/", "--")
    hub = types.ModuleType("huggingface_hub")
    hub.constants, hub.file_download = constants, file_download
    hub.snapshot_download = snapshot_download or (lambda *a, **k: (_ for _ in ()).throw(AssertionError("the probe must not ask the hub library to resolve a snapshot")))
    for name, mod in (("huggingface_hub", hub), ("huggingface_hub.constants", constants), ("huggingface_hub.file_download", file_download)):
        monkeypatch.setitem(sys.modules, name, mod)


def test_pinned_probe_inspects_the_snapshot_directory_itself(tmp_path, monkeypatch):
    """Regression for both directions of the hub's unreliable local-only resolution: it can return an empty/partial folder as
    usable, and raise IncompleteSnapshotError for a snapshot that merely lacks one file. The probe must not depend on it."""
    from retrieval_eval.retrievers import PINNED_MODEL_REVISION

    cache = tmp_path / "hub"
    _stub_hub(monkeypatch, cache)
    snap = cache / "models--sentence-transformers--all-MiniLM-L6-v2" / "snapshots" / PINNED_MODEL_REVISION
    assert runner_mod.pinned_snapshot_dir() == snap
    assert runner_mod.probe_pinned_model_cache()["state"] == "not_cached"          # no snapshot directory at all
    snap.mkdir(parents=True)
    empty = runner_mod.probe_pinned_model_cache()                                    # the originally reported case
    assert empty["state"] == "partial" and empty["files_present_with_expected_size"] == 0
    make_snapshot(snap, PINNED_MODEL_FILES)
    assert runner_mod.probe_pinned_model_cache()["state"] == "verified_complete"
    (snap / "modules.json").unlink()                                                 # one file missing: partial, NOT not_cached
    one = runner_mod.probe_pinned_model_cache()
    assert one["state"] == "partial" and one["missing"] == ["modules.json"] and one["files_present_with_expected_size"] == 10
    make_snapshot(snap, {"modules.json": PINNED_MODEL_FILES["modules.json"]})
    with open(snap / "model.safetensors", "wb") as fh:
        fh.truncate(1_000_000)                                                       # interrupted download
    cut = runner_mod.probe_pinned_model_cache()
    assert cut["state"] == "partial" and cut["size_mismatch"][0]["actual_bytes"] == 1_000_000


def test_pinned_probe_reports_unknown_when_the_cache_location_cannot_be_determined(monkeypatch):
    hub = types.ModuleType("huggingface_hub")  # has no `constants`/`file_download`: a changed library layout
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.delitem(sys.modules, "huggingface_hub.constants", raising=False)
    monkeypatch.delitem(sys.modules, "huggingface_hub.file_download", raising=False)
    res = runner_mod.probe_pinned_model_cache()
    assert res["state"] == "unknown" and "could not locate the hub cache" in res["detail"]
    assert describe_fetch(res["state"])[0] is True
