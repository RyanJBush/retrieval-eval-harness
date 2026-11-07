"""Export tests: JSON/CSV agreement, null-vs-zero, escaping, row counts, non-finite rejection, all-or-nothing writes,
tamper detection. Runs use FAKE retrievers and a controlled clock (tests/fakes.py)."""

import copy
import csv
import json
import math
import os
import statistics
from pathlib import Path

import pytest
from fakes import fake_run

from retrieval_eval import export
from retrieval_eval.data import load_dataset
from retrieval_eval.provenance import sha256_file

FIXTURE = Path(__file__).resolve().parents[1] / "data" / "synthetic"
FILES = ("documents.jsonl", "queries.jsonl", "qrels.csv")


@pytest.fixture(scope="module")
def dataset():
    return load_dataset(FIXTURE)


@pytest.fixture(scope="module")
def info():
    return {"directory": str(FIXTURE), "files": [{"name": f, "sha256": sha256_file(FIXTURE / f),
                                                  "bytes": (FIXTURE / f).stat().st_size} for f in FILES]}


@pytest.fixture(scope="module")
def run(dataset, info):
    return fake_run(dataset, FIXTURE, dataset_info=info)


@pytest.fixture
def written(tmp_path, run):
    export.write_run_artifacts(tmp_path, run)
    return tmp_path


# ------------------------------------------------------------------ strictness
def test_json_rejects_nan_and_infinity_and_writes_null():
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(export.ExportError, match="JSON"):
            export.dumps_strict({"x": [1, {"y": bad}]})
    text = export.dumps_strict({"a": None, "b": 0.0})
    assert json.loads(text) == {"a": None, "b": 0.0} and "NaN" not in text and "null" in text


def test_csv_cells_null_blank_zero_distinct_and_nonfinite_rejected():
    assert export._cell(None) == "" and export._cell(0.0) == "0.0" and export._cell(0) == "0"
    assert export._cell(0.1) == "0.1" and float(export._cell(1 / 3)) == 1 / 3  # repr round-trips exactly
    for bad in (float("nan"), float("inf"), -float("inf"), True, [1], {"a": 1}):
        with pytest.raises(export.ExportError):
            export._cell(bad)


def test_reading_json_with_nonstandard_constants_is_rejected(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text('{"x": NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-standard"):
        export.load_run(p)


# ------------------------------------------------------------------ row counts and columns
def test_default_artifact_row_counts(written):
    expected = {"summary.csv": 6, "per_query.csv": 48, "rankings.csv": 80, "latency.csv": 320,
                "latency_summary.csv": 18}
    for name, n in expected.items():
        header, rows = export.read_csv(written / name)
        assert len(rows) == n, name
        assert all(len(r) == len(header) for r in rows)
    assert export.read_csv(written / "latency.csv")[0] == ["method", "query_id", "repetition", "execution_index",
                                                           "position_in_pair", "elapsed_ns"]
    assert sorted(p.name for p in written.iterdir()) == sorted([*export.CSV_FILES, "run.json"])  # no temp leftovers


# ------------------------------------------------------------------ JSON <-> CSV agreement
def test_csvs_are_faithful_projections_of_run_json(written):
    run = export.load_run(written / "run.json")
    _, summary = export.read_csv(written / "summary.csv")
    for method, k, recall, mrr, total, scored, none in summary:
        agg = run["results"][method]["aggregate"][k]
        assert float(recall) == agg["macro_recall"] and float(mrr) == agg["mrr"]
        counts = run["results"][method]["counts"]
        assert (int(total), int(scored), int(none)) == (counts["total"], counts["scored"], counts["no_judged_relevant"])
    _, per_query = export.read_csv(written / "per_query.csv")
    for method, qid, text, k, status, n_rel, n_ret, recall, rr in per_query:
        rec = [q for q in run["results"][method]["queries"] if q["query_id"] == qid][0]
        met = rec["metrics"][k]
        assert status == rec["status"] and int(n_rel) == rec["n_relevant"] and int(n_ret) == rec["n_returned"]
        assert (recall == "") == (met["recall"] is None) and (rr == "") == (met["rr"] is None)
        if recall != "":
            assert float(recall) == met["recall"] and float(rr) == met["rr"]
    _, ranks = export.read_csv(written / "rankings.csv")
    for method, qid, rank, doc_id, score, label in ranks:
        rec = [q for q in run["results"][method]["queries"] if q["query_id"] == qid][0]["ranking"][int(rank) - 1]
        assert (rec["doc_id"], rec["label"], rec["score"]) == (doc_id, label, float(score))
    _, lat = export.read_csv(written / "latency.csv")
    assert [(r[0], r[1], int(r[2]), int(r[3]), int(r[4]), int(r[5])) for r in lat] == [
        (s["method"], s["query_id"], s["repetition"], s["execution_index"], s["position_in_pair"], s["elapsed_ns"])
        for s in run["latency"]["samples"]]


def test_aggregates_are_recomputable_from_exported_per_query_rows(written):
    _, per_query = export.read_csv(written / "per_query.csv")
    _, summary = export.read_csv(written / "summary.csv")
    for method, k, recall, mrr, total, scored, none in summary:
        rows = [r for r in per_query if r[0] == method and r[3] == k]
        assert len(rows) == int(total) == 8
        scored_rows = [r for r in rows if r[7] != ""]  # blank = no judged-relevant: excluded from the macro mean
        assert len(scored_rows) == int(scored) and len(rows) - len(scored_rows) == int(none)
        assert math.isclose(statistics.fmean(float(r[7]) for r in scored_rows), float(recall), abs_tol=1e-12)
        assert math.isclose(statistics.fmean(float(r[8]) for r in scored_rows), float(mrr), abs_tol=1e-12)


def test_saved_run_verifies_clean_including_dataset_hashes(written):
    run = export.load_run(written / "run.json")
    assert export.verify_saved_run(run, dataset_dir=FIXTURE) == []


# ------------------------------------------------------------------ null vs zero
def test_null_and_zero_are_distinguished_in_json_and_csv(tmp_path, dataset, info):
    scripted = {dataset.queries[0].text: []}  # q01 answerable but empty ranking -> zeros; q08 -> nulls
    run = fake_run(dataset, FIXTURE, dataset_info=info, per_method={"bm25": {"rankings": scripted}})
    export.write_run_artifacts(tmp_path, run)
    loaded = export.load_run(tmp_path / "run.json")
    q = {x["query_id"]: x for x in loaded["results"]["bm25"]["queries"]}
    assert q["q01"]["metrics"]["3"] == {"recall": 0.0, "rr": 0.0}
    assert q["q08"]["metrics"]["3"] == {"recall": None, "rr": None}
    _, rows = export.read_csv(tmp_path / "per_query.csv")
    cell = {(r[0], r[1], r[3]): (r[4], r[7], r[8]) for r in rows}
    assert cell[("bm25", "q01", "3")] == ("scored", "0.0", "0.0")  # zero is written as 0.0
    assert cell[("bm25", "q08", "3")] == ("no_judged_relevant", "", "")  # null is blank


def test_no_scorable_queries_gives_null_aggregates_not_zero(tmp_path):
    d = tmp_path / "ds"
    d.mkdir()
    (d / "documents.jsonl").write_text('{"doc_id": "a", "text": "x"}\n{"doc_id": "b", "text": "y"}\n', encoding="utf-8")
    (d / "queries.jsonl").write_text('{"query_id": "q", "text": "z"}\n', encoding="utf-8")
    (d / "qrels.csv").write_text("query_id,doc_id,relevance\nq,a,0\nq,b,0\n", encoding="utf-8")
    ds = load_dataset(d)
    run = fake_run(ds, d, repetitions=2, warmup=0)
    assert run["results"]["bm25"]["aggregate"]["1"] == {"macro_recall": None, "mrr": None}
    out = tmp_path / "out"
    out.mkdir()
    export.write_run_artifacts(out, run)
    _, rows = export.read_csv(out / "summary.csv")
    assert all(r[2] == "" and r[3] == "" and r[5] == "0" and r[6] == "1" for r in rows)  # blank, scored=0, none=1


# ------------------------------------------------------------------ escaping
def test_csv_escaping_roundtrips_commas_quotes_newlines_and_unicode(tmp_path):
    d = tmp_path / "ds"
    d.mkdir()
    q1 = 'He said "hi", then\nleft ☃ — café'
    q2 = "plain, with comma"
    (d / "documents.jsonl").write_text(
        json.dumps({"doc_id": 'd,"1"', "text": 'alpha, "beta"\nline2'}, ensure_ascii=False) + "\n"
        + json.dumps({"doc_id": "d2", "text": "second doc"}) + "\n", encoding="utf-8")
    (d / "queries.jsonl").write_text(
        json.dumps({"query_id": "q1", "text": q1}, ensure_ascii=False) + "\n"
        + json.dumps({"query_id": "q2", "text": q2}) + "\n", encoding="utf-8")
    (d / "qrels.csv").write_text('query_id,doc_id,relevance\nq1,"d,""1""",1\nq1,d2,0\nq2,"d,""1""",0\nq2,d2,1\n',
                                 encoding="utf-8")
    ds = load_dataset(d)
    run = fake_run(ds, d, repetitions=2, warmup=1)
    out = tmp_path / "out"
    out.mkdir()
    export.write_run_artifacts(out, run)
    _, rows = export.read_csv(out / "per_query.csv")
    assert {r[2] for r in rows} == {q1, q2}  # exact text, including the embedded newline
    _, ranks = export.read_csv(out / "rankings.csv")
    assert 'd,"1"' in {r[3] for r in ranks}
    raw = (out / "per_query.csv").read_text(encoding="utf-8")
    assert '"He said ""hi"", then\nleft ☃ — café"' in raw  # quoting is real CSV quoting, not ad-hoc
    loaded = export.load_run(out / "run.json")
    assert loaded["dataset"]["queries"][0]["text"] == q1 and loaded["dataset"]["documents"][0]["text"] == 'alpha, "beta"\nline2'
    assert export.verify_saved_run(loaded) == []


# ------------------------------------------------------------------ all-or-nothing writes
def test_nonfinite_value_in_run_writes_nothing(tmp_path, run):
    bad = copy.deepcopy(run)
    bad["results"]["bm25"]["queries"][0]["ranking"][0]["score"] = float("nan")
    with pytest.raises(export.ExportError):
        export.write_run_artifacts(tmp_path, bad)
    assert list(tmp_path.iterdir()) == []


def test_csv_write_failure_midway_leaves_no_files(tmp_path, run, monkeypatch):
    real, calls = export._write_csv, []

    def flaky(path, header, rows):
        calls.append(path.name)
        if len(calls) == 3:
            raise OSError("disk full")
        return real(path, header, rows)

    monkeypatch.setattr(export, "_write_csv", flaky)
    with pytest.raises(export.ExportError, match="disk full"):
        export.write_run_artifacts(tmp_path, run)
    assert list(tmp_path.iterdir()) == []


def test_run_json_is_written_last_so_failure_there_leaves_nothing(tmp_path, run, monkeypatch):
    real = os.replace

    def flaky(src, dst):
        if Path(dst).name == "run.json":
            raise OSError("cannot rename run.json")
        return real(src, dst)

    monkeypatch.setattr(os, "replace", flaky)
    with pytest.raises(export.ExportError):
        export.write_run_artifacts(tmp_path, run)
    assert list(tmp_path.iterdir()) == []  # the five CSVs that were already in place were removed again


# ------------------------------------------------------------------ verification detects tampering
@pytest.mark.parametrize("mutate,needle", [
    (lambda r: r["results"]["bm25"]["queries"][0]["metrics"]["3"].__setitem__("recall", 0.123), "saved metrics differ"),
    (lambda r: r["results"]["embedding"]["aggregate"]["1"].__setitem__("mrr", 0.999), "aggregate differs"),
    (lambda r: r["results"]["bm25"]["counts"].__setitem__("scored", 8), "counts"),
    (lambda r: r["results"]["bm25"]["queries"][0]["ranking"][0].__setitem__("label", "unjudged"), "label"),
    (lambda r: r["latency"]["samples"].pop(), "latency samples"),
    (lambda r: r["latency"]["summary"]["bm25"]["overall"].__setitem__("median_ns", 1.0), "overall latency"),
    (lambda r: r["latency"]["summary"]["embedding"]["per_query"]["q02"].__setitem__("median_ns", 1.0), "per-query latency"),
    (lambda r: r["run"].__setitem__("status", "incomplete"), "status"),
    (lambda r: r.__setitem__("schema_version", "9"), "schema_version"),
])
def test_verify_saved_run_flags_inconsistencies(run, mutate, needle):
    bad = copy.deepcopy(run)
    mutate(bad)
    problems = export.verify_saved_run(bad)
    assert problems and any(needle in p for p in problems), problems


def test_verify_detects_changed_dataset_file(run, tmp_path):
    import shutil

    d = tmp_path / "ds"
    shutil.copytree(FIXTURE, d)
    with open(d / "queries.jsonl", "a", encoding="utf-8") as fh:
        fh.write("\n")
    assert any("queries.jsonl" in p for p in export.verify_saved_run(run, dataset_dir=d))


# ------------------------------------------------------------------ comparing two runs
def test_compare_runs_identical_inputs_differ_only_in_run_specifics(dataset, info):
    a = fake_run(dataset, FIXTURE, dataset_info=info)
    b = fake_run(dataset, FIXTURE, dataset_info=info)
    b["run"]["run_id"], b["run"]["started_utc"] = "other", "2000-01-01T00:00:00+00:00"
    b["latency"]["samples"][0]["elapsed_ns"] += 1
    result = export.compare_runs(a, b)
    assert result["identity_equal"] and result["rankings_equal"] and result["quality_equal"]
    assert result["max_abs_score_difference"] == 0.0
    assert result["run_ids_differ"] and result["started_differs"] and not result["latencies_identical"]


def test_compare_runs_flags_ranking_quality_and_identity_changes(dataset, info):
    a = fake_run(dataset, FIXTURE, dataset_info=info)
    other = fake_run(dataset, FIXTURE, dataset_info=info,
                     per_method={"bm25": {"rankings": {dataset.queries[0].text: [("d05", 1.0), ("d01", 0.5)]}}})
    result = export.compare_runs(a, other)
    assert not result["rankings_equal"] and "bm25/q01" in result["ranking_differences"]
    assert not result["quality_equal"] and "bm25/aggregate" in result["quality_differences"]
    c = fake_run(dataset, FIXTURE, dataset_info=info, repetitions=5)
    assert "config" in export.compare_runs(a, c)["identity_differences"]


def test_verifier_rejects_a_completed_run_that_records_ranking_disagreements(run):
    bad = copy.deepcopy(run)
    bad["ranking_consistency"]["ranking_disagreements"] = 1
    assert any("ranking disagreements" in p for p in export.verify_saved_run(bad))
    bad = copy.deepcopy(run)
    bad["results"]["bm25"]["queries"][0]["measured_search_consistency"]["n_ranking_identical"] -= 1
    assert any("different ordered ID list" in p for p in export.verify_saved_run(bad))
    bad = copy.deepcopy(run)
    del bad["ranking_consistency"]
    assert any("ranking_consistency" in p for p in export.verify_saved_run(bad))


@pytest.mark.parametrize("mutate,needle", [
    (lambda r: r["results"]["embedding"]["queries"][1]["ranking"][0].__setitem__("rank", 99), "expected rank 1, got 99"),
    (lambda r: r["results"]["bm25"]["queries"][0]["ranking"][1].__setitem__("rank", 1), "expected rank 2, got 1"),
    (lambda r: r["results"]["bm25"]["queries"][0]["ranking"][0].__setitem__("rank", True), "expected rank 1, got True"),
    (lambda r: r["results"]["bm25"]["queries"][0]["ranking"][0].__setitem__("rank", 1.0), "expected rank 1, got 1.0"),
    (lambda r: r["results"]["bm25"]["queries"][0].__setitem__("n_returned", 3), "n_returned 3"),
])
def test_verifier_checks_ranks_against_list_positions_and_n_returned(run, mutate, needle):
    bad = copy.deepcopy(run)
    mutate(bad)
    problems = export.verify_saved_run(bad)
    assert any(needle in p for p in problems), problems


@pytest.mark.parametrize("version", ["1.0", "1.1"])
def test_verifier_accepts_valid_ranks_in_both_schema_versions(run, version):
    assert export.verify_saved_run(dict(run, schema_version=version)) == []
