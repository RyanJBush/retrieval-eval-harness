"""Viewer tests. Runs come from the real runner over FAKE retrievers (tests/fakes.py); the scripted rankings for q02, q03 and
q07 copy the SHAPE of the real run (BM25 misses d10 for q02; only the embedding finds the second relevant document for q03
and q07). Nothing here needs the dataset directory, a model, or the network."""

import copy
import html.parser
import http.client
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from fakes import fake_run

from retrieval_eval import cli, export, viewer
from retrieval_eval.data import load_dataset
from retrieval_eval.provenance import sha256_file

FIXTURE = Path(__file__).resolve().parents[1] / "data" / "synthetic"
DS = load_dataset(FIXTURE)
TEXT = {q.query_id: q.text for q in DS.queries}
QIDS = list(DS.query_ids)
KS = (1, 3, 5)


def scored(ids_scores):
    return [(d, s) for d, s in ids_scores]


BM25 = {
    TEXT["q01"]: scored([("d01", 2.2923), ("d02", 0.0), ("d03", 0.0), ("d04", 0.0), ("d05", 0.0)]),
    TEXT["q02"]: scored([("d11", 1.6263), ("d12", 1.6263), ("d20", 1.5848), ("d17", 0.6445), ("d09", 0.6267)]),
    TEXT["q03"]: scored([("d03", 5.7228), ("d05", 0.4112), ("d04", 0.4002), ("d16", 0.4002), ("d10", 0.3897)]),
    TEXT["q07"]: scored([("d18", 2.7647), ("d05", 0.4112), ("d03", 0.4002), ("d04", 0.4002), ("d16", 0.4002)]),
}
EMB = {
    TEXT["q01"]: scored([("d01", 0.2708), ("d17", 0.2323), ("d04", 0.1269), ("d11", 0.1016), ("d16", 0.0991)]),
    TEXT["q02"]: scored([("d10", 0.5772), ("d09", 0.3635), ("d01", 0.1810), ("d15", 0.1712), ("d17", 0.1288)]),
    TEXT["q03"]: scored([("d03", 0.6695), ("d02", 0.5104), ("d12", 0.1626), ("d10", 0.1521), ("d01", 0.1090)]),
    TEXT["q07"]: scored([("d18", 0.7346), ("d19", 0.3336), ("d01", 0.2176), ("d05", 0.1362), ("d03", 0.1191)]),
}


@pytest.fixture(scope="module")
def run():
    info = {"directory": str(FIXTURE), "files": [{"name": f, "sha256": sha256_file(FIXTURE / f), "bytes": 1}
                                                 for f in ("documents.jsonl", "queries.jsonl", "qrels.csv")]}
    return fake_run(DS, FIXTURE, dataset_info=info, per_method={"bm25": {"rankings": BM25}, "embedding": {"rankings": EMB}})


@pytest.fixture
def run_file(tmp_path, run):
    p = tmp_path / "copied" / "run.json"
    p.parent.mkdir()
    p.write_text(export.dumps_strict(run), encoding="utf-8")
    return p


def method(view, name):
    return next(m for m in view["methods"] if m["method"] == name)


# ------------------------------------------------------------------ loading: valid and rejected runs
def test_valid_saved_run_loads_from_a_copied_file(run_file, run):
    loaded = viewer.load_for_view(run_file)
    assert loaded["run"]["run_id"] == run["run"]["run_id"]


def test_schema_1_0_runs_remain_viewable(tmp_path, run):
    p = tmp_path / "old.json"
    p.write_text(export.dumps_strict(dict(run, schema_version="1.0")), encoding="utf-8")
    assert viewer.load_for_view(p)["schema_version"] == "1.0"


def readable(exc: viewer.ViewerError) -> str:
    msg = str(exc)
    assert msg and not any(bad in msg for bad in ("Traceback", "KeyError", "TypeError", "AttributeError", "object at 0x"))
    return msg


def reject(path, needle):
    with pytest.raises(viewer.ViewerError) as exc:
        viewer.load_for_view(path)
    msg = readable(exc.value)
    assert needle.lower() in msg.lower(), msg
    return msg


def test_rejects_missing_empty_garbage_truncated_and_non_utf8_files(tmp_path, run_file):
    reject(tmp_path / "nope.json", "file not found")
    for name, content, needle in [("empty.json", b"", "not valid JSON"), ("garbage.json", b"hello world", "not valid JSON"),
                                  ("truncated.json", run_file.read_bytes()[:300], "not valid JSON"),
                                  ("bin.json", b"\xff\xfe\x00\x01", "UTF-8"), ("array.json", b"[1, 2]", "JSON object")]:
        p = tmp_path / name
        p.write_bytes(content)
        reject(p, needle)


def test_rejects_nonstandard_json_constants(tmp_path, run):
    text = export.dumps_strict(run).replace('"threads_requested": 1', '"threads_requested": NaN')
    p = tmp_path / "nan.json"
    p.write_text(text, encoding="utf-8")
    reject(p, "not a valid saved run")


def test_rejects_a_failure_manifest_as_an_incomplete_run(tmp_path):
    out = tmp_path / "o"
    q01 = TEXT["q01"]
    from fakes import FakeClock, make_builders

    clock = FakeClock()
    rc = cli.main(["--dataset", str(FIXTURE), "--out-dir", str(out)],
                  builders=make_builders(clock, DS.doc_ids, per_method={"bm25": {"fail": {(q01, 1): ValueError("boom")}}}),
                  clock=clock)
    assert rc == 1
    msg = reject(out / "run_failed.json", "INCOMPLETE run")
    assert "stage=first_call" in msg and "method=bm25" in msg and "boom" in msg


def test_rejects_runs_not_completed_and_unsupported_schema(tmp_path, run):
    bad = copy.deepcopy(run)
    bad["run"]["status"] = "running"
    p = tmp_path / "status.json"
    p.write_text(json.dumps(bad), encoding="utf-8")
    reject(p, "not 'completed'")
    for version in ("2.0", "0.9", None, 1.1):
        bad = dict(run, schema_version=version)
        p.write_text(json.dumps(bad), encoding="utf-8")
        reject(p, "unsupported run schema_version")


@pytest.mark.parametrize("mutate,needle", [
    (lambda r: r["results"].pop("bm25"), "results.bm25"),
    (lambda r: r["dataset"].pop("documents"), "dataset.documents"),
    (lambda r: r["latency"]["summary"].pop("embedding"), "latency.summary.embedding"),
    (lambda r: r["config"].__setitem__("ks", "1,3,5"), "config.ks"),
    (lambda r: r["results"]["bm25"]["queries"][0].pop("ranking"), "ranking"),
    (lambda r: r["results"]["bm25"]["queries"][0]["ranking"][0].__setitem__("score", "high"), "score"),
    (lambda r: r["dataset"]["queries"].clear(), "no queries"),
    (lambda r: r.__setitem__("results", []), "results"),
])
def test_malformed_runs_are_rejected_with_the_offending_path(tmp_path, run, mutate, needle):
    bad = copy.deepcopy(run)
    mutate(bad)
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(bad), encoding="utf-8")
    reject(p, needle)


def test_inconsistent_run_is_rejected_by_the_integrity_checks(tmp_path, run):
    bad = copy.deepcopy(run)
    bad["results"]["embedding"]["queries"][1]["metrics"]["3"]["recall"] = 0.123
    p = tmp_path / "tampered.json"
    p.write_text(json.dumps(bad), encoding="utf-8")
    msg = reject(p, "integrity checks")
    assert "embedding/q02" in msg


@pytest.mark.parametrize("version", ["1.0", "1.1"])
def test_valid_run_opens_in_both_schema_versions_with_the_rank_check(tmp_path, run, version):
    p = tmp_path / "ok.json"
    p.write_text(json.dumps(dict(run, schema_version=version)), encoding="utf-8")
    assert viewer.load_for_view(p)["schema_version"] == version


def _set_rank(r, qi, ri, value):
    r["results"]["embedding"]["queries"][qi]["ranking"][ri]["rank"] = value


@pytest.mark.parametrize("version", ["1.0", "1.1"])
def test_the_reported_malformed_copy_is_rejected(tmp_path, run, version):
    """The exact reviewer reproduction: only the rank of the first embedding result for q02 changed, 1 -> 99."""
    bad = dict(copy.deepcopy(run), schema_version=version)
    assert bad["results"]["embedding"]["queries"][1]["query_id"] == "q02"
    _set_rank(bad, 1, 0, 99)
    p = tmp_path / "wrong_rank.json"
    p.write_text(json.dumps(bad), encoding="utf-8")
    msg = reject(p, "expected rank 1")
    assert "embedding/q02" in msg and "99" in msg


@pytest.mark.parametrize("mutate,needle", [
    (lambda r: _set_rank(r, 1, 0, 0), "expected rank 1"),                       # zero-based
    (lambda r: _set_rank(r, 1, 0, 2), "expected rank 1"),                       # duplicate of rank 2
    (lambda r: _set_rank(r, 1, 2, 7), "expected rank 3"),                       # gap in the middle
    (lambda r: _set_rank(r, 1, 0, True), "expected rank 1"),                    # bool is not an integer rank
    (lambda r: _set_rank(r, 1, 0, 1.0), "ranking[].rank' should be int"),                # float rejected by the structure check
    (lambda r: _set_rank(r, 1, 0, "1"), "ranking[].rank' should be int"),
    (lambda r: _set_rank(r, 1, 0, None), "ranking[].rank' should be int"),
    (lambda r: r["results"]["embedding"]["queries"][1]["ranking"][0].pop("rank"), "rank"),
    (lambda r: r["results"]["bm25"]["queries"][1]["ranking"].reverse(), "expected rank"),  # list order no longer matches the ranks
    (lambda r: r["results"]["bm25"]["queries"][1].__setitem__("n_returned", 4), "n_returned 4"),
])
def test_ranks_must_equal_consecutive_one_based_list_positions(tmp_path, run, mutate, needle):
    bad = copy.deepcopy(run)
    mutate(bad)
    p = tmp_path / "bad_rank.json"
    p.write_text(json.dumps(bad), encoding="utf-8")
    msg = reject(p, needle)
    assert "Traceback" not in msg


def test_a_rank_change_cannot_make_the_viewer_contradict_the_saved_metrics(tmp_path, run):
    """What the bug allowed: Recall@5 of 1.0 shown next to 'missed at k=5'. The run must now be refused outright."""
    bad = copy.deepcopy(run)
    _set_rank(bad, 1, 0, 99)
    p = tmp_path / "contradiction.json"
    p.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(viewer.ViewerError):
        viewer.load_for_view(p)


@pytest.mark.parametrize("shape", [
    {"status": "incomplete", "errors": {"stage": "build"}},                       # the reported file: errors is an object
    {"status": "incomplete", "errors": "boom"},
    {"status": "incomplete", "errors": None},
    {"status": "incomplete", "errors": []},
    {"status": "incomplete", "errors": ["just a string"]},
    {"status": "incomplete", "errors": [None, {"stage": "build"}]},
    {"status": "incomplete", "errors": [{"stage": 5, "method": ["x"], "query_id": {"a": 1}, "message": {"m": 1}}]},
    {"status": "incomplete"},
])
def test_malformed_failure_manifests_exit_2_with_a_short_readable_error_and_no_server(tmp_path, monkeypatch, capsys, shape):
    def must_not_serve(*a, **k):
        raise AssertionError("a server was created for an incomplete run")

    monkeypatch.setattr(viewer, "make_server", must_not_serve)
    p = tmp_path / "run_failed.json"
    p.write_text(json.dumps(shape), encoding="utf-8")
    assert viewer.main([str(p)]) == 2
    out = capsys.readouterr()
    assert out.out == "" and out.err.startswith("error: this is an INCOMPLETE run")
    assert len(out.err) < 400 and "Traceback" not in out.err and "KeyError" not in out.err


def test_the_reported_failure_manifest_in_a_real_process(tmp_path):
    p = tmp_path / "run_failed.json"
    p.write_text('{"status":"incomplete","errors":{"stage":"build"}}', encoding="utf-8")
    proc = subprocess.run([sys.executable, "-m", "retrieval_eval.viewer", str(p)], capture_output=True, text=True, timeout=30)
    assert proc.returncode == 2 and proc.stdout == "" and "INCOMPLETE run" in proc.stderr and "Traceback" not in proc.stderr


def _delete_paths(node, prefix=(), depth=0):
    """Every dict key (to depth 4, first list element only) as a path of keys/indices, for a delete-one-key sweep."""
    if depth > 4:
        return
    if isinstance(node, dict):
        for k, v in node.items():
            yield prefix + (k,)
            yield from _delete_paths(v, prefix + (k,), depth + 1)
    elif isinstance(node, list) and node:
        yield from _delete_paths(node[0], prefix + (0,), depth + 1)


def test_deleting_any_single_key_never_escapes_as_a_raw_exception(tmp_path, run):
    skip_prefix = {("latency", "samples"), ("latency", "warmup_samples"), ("dependencies",)}  # large, irrelevant lists/maps
    paths = [p for p in _delete_paths(run) if not any(p[:len(s)] == s and len(p) > len(s) for s in skip_prefix)]
    assert len(paths) > 150
    p = tmp_path / "mut.json"
    outcomes = {"loaded": 0, "rejected": 0}
    for path in paths:
        bad = copy.deepcopy(run)
        node = bad
        for step in path[:-1]:
            node = node[step]
        del node[path[-1]]
        p.write_text(json.dumps(bad), encoding="utf-8")
        try:
            viewer.load_for_view(p)
            outcomes["loaded"] += 1  # an optional field (e.g. model, environment details) was removed: still viewable
        except viewer.ViewerError as exc:
            readable(exc)
            outcomes["rejected"] += 1
    assert outcomes["rejected"] > 50 and outcomes["loaded"] >= 1  # only ViewerError ever escaped (any other exception fails)


def test_optional_sections_missing_still_renders(tmp_path, run):
    slim = copy.deepcopy(run)
    for key in ("model", "environment", "timing_protocol"):
        slim.pop(key, None)
    p = tmp_path / "slim.json"
    p.write_text(json.dumps(slim), encoding="utf-8")
    page = viewer.render_page(viewer.load_for_view(p))
    assert "unknown" in page and "Saved retrieval run" in page
    # the repeat-search consistency block is NOT optional: without it the run cannot show it had no ranking disagreement
    del slim["ranking_consistency"]
    p.write_text(json.dumps(slim), encoding="utf-8")
    reject(p, "ranking_consistency")


# ------------------------------------------------------------------ display model
def test_overview_matches_the_saved_run(run):
    ov = viewer.build_overview(run)
    assert ov["run_id"] == run["run"]["run_id"] and ov["designation"] == "synthetic" and "not a benchmark" in ov["designation_note"]
    assert len(ov["aggregate_rows"]) == 6
    for r in ov["aggregate_rows"]:
        agg = run["results"][r["method"]]["aggregate"][str(r["k"])]
        assert (r["macro_recall"], r["mrr"]) == (agg["macro_recall"], agg["mrr"])
        assert (r["scored"], r["total"], r["no_judged_relevant"]) == (7, 8, 1)
    for m, v in ov["latency"].items():
        o = run["latency"]["summary"][m]["overall"]
        assert v["n"] == 160 == o["n"] and v["median_ms"] == o["median_ns"] / 1e6 and v["p95_ms"] == o["p95_ns"] / 1e6
    assert "Warm, local, serial CPU" in ov["conditions"] and "3 discarded warm-up" in ov["conditions"]
    assert "20 measured searches" in ov["conditions"] and "1 thread" in ov["conditions"] and "not independent" in ov["conditions"]


def test_external_designation_is_not_labeled_synthetic(run):
    ext = copy.deepcopy(run)
    ext["dataset"]["designation"] = "external"
    ov = viewer.build_overview(ext)
    assert ov["designation"] == "external" and "not a benchmark" not in ov["designation_note"]


def expected_outcome(run, qid, method_name, k):
    """Independent recomputation from judgments + saved ranking (not using the viewer's code)."""
    rel = {d for q, d, r in run["dataset"]["judgments"] if q == qid and r == 1}
    ranking = next(q for q in run["results"][method_name]["queries"] if q["query_id"] == qid)["ranking"]
    pos = {r["doc_id"]: r["rank"] for r in ranking}
    return {d: ("found" if d in pos and pos[d] <= k else "missed") for d in rel}, pos


@pytest.mark.parametrize("qid", QIDS)
@pytest.mark.parametrize("k", KS)
def test_every_query_and_cutoff_is_viewable_and_consistent(run, qid, k):
    view = viewer.build_query_view(run, qid, k)
    assert view["query_id"] == qid and view["text"] == TEXT[qid] and view["k"] == k
    assert [m["method"] for m in view["methods"]] == ["bm25", "embedding"]
    qrels = {d: r for q, d, r in run["dataset"]["judgments"] if q == qid}
    for m in view["methods"]:
        saved = next(q for q in run["results"][m["method"]]["queries"] if q["query_id"] == qid)
        assert [r["rank"] for r in m["rows"]] == list(range(1, min(k, saved["n_returned"]) + 1))
        for row, s in zip(m["rows"], saved["ranking"]):
            want = {1: "relevant", 0: "nonrelevant", None: "unjudged"}[qrels.get(row["doc_id"])]
            assert (row["doc_id"], row["score"], row["label"]) == (s["doc_id"], s["score"], want)
            assert row["text"] == next(d["text"] for d in run["dataset"]["documents"] if d["doc_id"] == row["doc_id"])
        outcome, pos = expected_outcome(run, qid, m["method"], k)
        assert {d["doc_id"]: d["outcome"] for d in m["relevant_documents"]} == outcome
        met = saved["metrics"][str(k)]
        assert (m["recall"], m["rr"]) == (met["recall"], met["rr"])
        if outcome:
            assert m["recall"] == sum(v == "found" for v in outcome.values()) / len(outcome)  # misses explain the recall


def test_unscored_query_is_distinct_from_zero_recall(run):
    q08 = viewer.build_query_view(run, "q08", 3)
    for m in q08["methods"]:
        assert m["verdict"] == "unscored" and m["recall"] is None and m["rr"] is None and m["relevant_documents"] == []
        assert "not zero" in m["note"]
    q02 = viewer.build_query_view(run, "q02", 5)
    bm25, emb = method(q02, "bm25"), method(q02, "embedding")
    assert bm25["verdict"] == "zero_recall" and bm25["recall"] == 0.0 and "not an unscored query" in bm25["note"]
    assert emb["verdict"] == "scored" and emb["recall"] == 1.0 and emb["note"] is None


def test_q02_demo_path_bm25_misses_the_relevant_document_embedding_ranks_it_first(run):
    view = viewer.build_query_view(run, "q02", 5)
    bm25, emb = method(view, "bm25"), method(view, "embedding")
    assert [r["doc_id"] for r in bm25["rows"]] == ["d11", "d12", "d20", "d17", "d09"]
    assert bm25["relevant_documents"] == [{"doc_id": "d10", "outcome": "missed", "detail": "missed: not in the saved top-5",
                                           "text": next(d["text"] for d in run["dataset"]["documents"] if d["doc_id"] == "d10")}]
    assert emb["rows"][0]["doc_id"] == "d10" and emb["rows"][0]["label"] == "relevant"
    assert emb["relevant_documents"][0]["detail"] == "found at rank 1"


def test_q03_and_q07_demo_paths_first_relevant_at_rank_one_second_only_for_embedding(run):
    for qid, second, name_other in (("q03", "d02", "bm25"), ("q07", "d19", "bm25")):
        v5 = viewer.build_query_view(run, qid, 5)
        bm25, emb = method(v5, "bm25"), method(v5, "embedding")
        assert bm25["rows"][0]["label"] == emb["rows"][0]["label"] == "relevant"  # first relevant at rank one, both
        assert bm25["recall"] == 0.5 and bm25["rr"] == 1.0 and emb["recall"] == 1.0 and emb["rr"] == 1.0
        assert {d["doc_id"]: d["outcome"] for d in bm25["relevant_documents"]}[second] == "missed"
        assert [d["detail"] for d in bm25["relevant_documents"] if d["doc_id"] == second] == ["missed: not in the saved top-5"]
        assert {d["doc_id"]: d["outcome"] for d in emb["relevant_documents"]}[second] == "found"
        v1 = viewer.build_query_view(run, qid, 1)  # at k=1 even the embedding has not found it yet; it is at rank 2 of top-5
        assert method(v1, "embedding")["recall"] == 0.5
        assert [d["detail"] for d in method(v1, "embedding")["relevant_documents"] if d["doc_id"] == second] == [
            "missed at k=1: it is at rank 2 of the saved top-5"]


def test_q01_exact_identifier_is_found_by_both_at_rank_one(run):
    v = viewer.build_query_view(run, "q01", 1)
    assert all(m["rows"][0]["doc_id"] == "d01" and m["rows"][0]["label"] == "relevant" for m in v["methods"])


def test_invalid_selection_is_a_viewer_error(run):
    for qid, k in (("q99", 3), ("q01", 4), ("q01", 0)):
        with pytest.raises(viewer.ViewerError):
            viewer.build_query_view(run, qid, k)


# ------------------------------------------------------------------ HTML rendering
def test_page_shows_header_tables_selectors_and_method_specific_score_labels(run):
    page = viewer.render_page(run, "q02", 3)
    assert run["run"]["run_id"] in page and "Dataset designation: synthetic" in page
    for qid in QIDS:
        assert f'<option value="{qid}"' in page
    assert '<option value="q02" selected>' in page and '<option value="3" selected>' in page
    for k in KS:
        assert f'<option value="{k}"' in page
    assert "score (bm25: BM25 score, unbounded; larger is better)" in page
    assert "score (embedding: cosine similarity, between -1 and 1; larger is better)" in page
    assert "must not be compared across the two columns" in page
    assert "median (ms)" in page and "p95 (ms)" in page and "n (measured searches)" in page
    assert "Warm, local, serial CPU" in page
    assert "Scored, zero recall" in page  # the q02 bm25 miss
    assert "d10: missed: not in the saved top-5" in page and "d10: found at rank 1" in page
    assert '<span class="badge relevant">relevant</span>' in page and '<span class="badge nonrelevant">nonrelevant</span>' in page


def test_unscored_query_page_says_unscored_not_zero(run):
    page = viewer.render_page(run, "q08", 5)
    assert "Unscored: this query has no judged-relevant document" in page and "Recall@5: <strong>n/a</strong>" in page
    assert "Scored, zero recall" not in page and "none (no judged-relevant document for this query)" in page


def test_no_results_returned_for_an_answerable_query_is_shown(dataset_run=None):
    empty = fake_run(DS, FIXTURE, repetitions=2, warmup=0, per_method={"bm25": {"rankings": {TEXT["q01"]: []}}})
    page = viewer.render_page(empty, "q01", 3)
    assert "no results returned for this query" in page and "Scored, zero recall" in page


def test_everything_dynamic_is_html_escaped(run):
    evil = copy.deepcopy(run)
    evil["dataset"]["documents"][0]["text"] = '<script>alert("x")</script> & <b>bold</b>'
    evil["dataset"]["queries"][0]["text"] = '"><img src=x onerror=alert(1)>'
    evil["run"]["run_id"] = "<svg onload=alert(1)>"
    evil["dataset"]["designation"] = "synthetic"
    page = viewer.render_page(evil, "q01", 5)
    assert "<script" not in page and "<img" not in page and "<svg" not in page and "<b>" not in page
    assert "&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt; &amp; &lt;b&gt;bold&lt;/b&gt;" in page
    assert "&lt;img src=x onerror=alert(1)&gt;" in page and "&lt;svg onload=alert(1)&gt;" in page
    err = viewer.render_page(run, error="bad <script>alert(1)</script> input")
    assert "<script" not in err and "&lt;script&gt;" in err


# ------------------------------------------------------------------ the local server
@pytest.fixture
def server(run):
    srv, base = viewer.serve_in_thread(run)
    yield srv, base
    srv.shutdown()
    srv.server_close()


def get(srv, path="/", method="GET", host=None):
    conn = http.client.HTTPConnection("127.0.0.1", srv.server_port, timeout=10)
    headers = {"Host": host} if host else {}
    conn.request(method, path, headers=headers)
    resp = conn.getresponse()
    body = resp.read().decode("utf-8")
    out = (resp.status, dict(resp.getheaders()), body)
    conn.close()
    return out


def test_server_binds_to_loopback_only(server):
    srv, base = server
    assert srv.server_address[0] == "127.0.0.1" and base.startswith("http://127.0.0.1:")


def test_server_serves_every_query_and_cutoff(server, run):
    srv, _ = server
    status, headers, body = get(srv)
    assert status == 200 and run["run"]["run_id"] in body and '<option value="q01" selected>' in body and '<option value="5" selected>' in body
    for qid in QIDS:
        for k in KS:
            status, _, body = get(srv, f"/?q={qid}&k={k}")
            assert status == 200 and f'<option value="{qid}" selected>' in body and f'<option value="{k}" selected>' in body
    status, _, body = get(srv, "/?q=q02")  # k defaults to the largest saved cutoff
    assert status == 200 and '<option value="5" selected>' in body


def test_server_security_headers_and_content_type(server):
    srv, _ = server
    _, headers, _ = get(srv)
    assert headers["Content-Type"] == "text/html; charset=utf-8" and headers["Cache-Control"] == "no-store"
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert "default-src 'none'" in headers["Content-Security-Policy"] and "script-src" not in headers["Content-Security-Policy"]


@pytest.mark.parametrize("path", ["/?q=q99", "/?q=%3Cscript%3E", "/?k=0", "/?k=4", "/?k=abc", "/?k=3.0", "/?k=-1", "/?q=q01&k=%EF%BC%93", "/?q="])
def test_invalid_selection_gives_a_readable_400_page_with_escaped_input(server, path):
    srv, _ = server
    status, _, body = get(srv, path)
    assert status == 400 and 'class="error"' in body and "Saved retrieval run" in body
    assert "<script" not in body.replace("<style>", "")


def test_unknown_paths_methods_and_hosts(server):
    srv, _ = server
    for p in ("/other", "/../etc/passwd", "/run.json", "/%2e%2e/secret"):
        assert get(srv, p)[0] == 404
    for m in ("POST", "PUT", "DELETE", "HEAD", "OPTIONS"):
        status, headers, _ = get(srv, "/", method=m)
        assert status == 405 and headers.get("Allow") == "GET"
    assert get(srv, "/", host="evil.example.com")[0] == 421
    assert get(srv, "/", host=f"localhost:{srv.server_port}")[0] == 200
    assert get(srv, "/", host="127.0.0.1:1")[0] == 421  # right host name, wrong port


# ------------------------------------------------------------------ command line
def test_export_html_writes_one_page(tmp_path, run_file, capsys):
    out = tmp_path / "page.html"
    assert viewer.main([str(run_file), "--export-html", str(out), "--query", "q03", "--k", "5"]) == 0
    page = out.read_text(encoding="utf-8")
    assert '<option value="q03" selected>' in page and "d02: missed: not in the saved top-5" in page
    assert "wrote" in capsys.readouterr().out


@pytest.mark.parametrize("extra,needle", [(["--query", "q99"], "unknown query"), (["--k", "4"], "invalid cutoff")])
def test_export_html_rejects_bad_selection_with_exit_2(tmp_path, run_file, capsys, extra, needle):
    assert viewer.main([str(run_file), "--export-html", str(tmp_path / "p.html"), *extra]) == 2
    assert needle in capsys.readouterr().err and not (tmp_path / "p.html").exists()


def test_cli_rejects_unreadable_and_invalid_inputs_with_exit_2(tmp_path, capsys):
    assert viewer.main([str(tmp_path / "missing.json")]) == 2
    assert "file not found" in capsys.readouterr().err
    bad = tmp_path / "bad.json"
    bad.write_text("{", encoding="utf-8")
    assert viewer.main([str(bad)]) == 2 and "not valid JSON" in capsys.readouterr().err
    assert viewer.main([]) == 2  # argparse usage error
    assert viewer.main([str(bad), "--port", "70000"]) == 2


def test_cli_port_in_use_is_a_readable_error_and_cannot_hang(run_file, server, monkeypatch, capsys):
    """Bounded failure path: if a bind on an occupied port ever succeeded, serve_forever would run for ever and hang CI.
    Here serve_forever raises instead, so such a regression fails immediately rather than blocking the test run."""
    def must_not_serve(self, *a, **k):
        raise AssertionError("the viewer started serving on a port that already has a live listener")

    monkeypatch.setattr(viewer._ViewerServer, "serve_forever", must_not_serve)
    srv, _ = server
    assert viewer.main([str(run_file), "--port", str(srv.server_port)]) == 2
    err = capsys.readouterr().err
    assert "cannot start the viewer" in err and f"127.0.0.1:{srv.server_port}" in err and "--port 0" in err


def test_cli_port_in_use_in_a_real_process_exits_2_within_a_timeout(run_file, server):
    srv, _ = server
    proc = subprocess.run([sys.executable, "-m", "retrieval_eval.viewer", str(run_file), "--port", str(srv.server_port)],
                          capture_output=True, text=True, timeout=30)  # a hang raises TimeoutExpired: a bounded failure
    assert proc.returncode == 2 and "cannot start the viewer" in proc.stderr and "Traceback" not in proc.stderr


def test_second_server_cannot_bind_a_port_with_a_live_listener(run, server):
    srv, _ = server
    try:
        second = viewer.make_server(run, srv.server_port)
    except OSError:
        return  # the required behavior; no serving loop was ever started, so this cannot hang
    second.server_close()
    pytest.fail("two viewers bound the same loopback port")


def test_windows_binding_is_made_exclusive_and_other_platforms_are_untouched():
    import socket
    import types

    calls = []
    sock = types.SimpleNamespace(setsockopt=lambda *a: calls.append(a))
    for platform in ("linux", "darwin", "freebsd"):
        viewer.apply_exclusive_bind(sock, platform)
    assert calls == [] and all(viewer.reuse_address_allowed(p) for p in ("linux", "darwin", "freebsd"))
    viewer.apply_exclusive_bind(sock, "win32")
    assert calls == [(socket.SOL_SOCKET, getattr(socket, "SO_EXCLUSIVEADDRUSE", -5), 1)]  # exclusive use switched on
    assert viewer.reuse_address_allowed("win32") is False  # and SO_REUSEADDR (shared binding) switched off


def test_server_bind_applies_the_exclusive_option_to_its_own_socket_before_binding(run, monkeypatch):
    seen = []
    real = viewer.apply_exclusive_bind
    monkeypatch.setattr(viewer, "apply_exclusive_bind", lambda sock, *a: (seen.append(sock), real(sock, *a)))
    srv = viewer.make_server(run, 0)
    try:
        assert seen == [srv.socket] and viewer._ViewerServer.allow_reuse_address == viewer.reuse_address_allowed()
        assert srv.server_address[0] == "127.0.0.1" and srv.server_port > 0  # still loopback-only, port 0 still works
    finally:
        srv.server_close()


@pytest.mark.skipif(sys.platform == "win32", reason="SIGINT delivery to a child process differs on Windows")
def test_ctrl_c_stops_a_running_viewer_with_exit_code_0(run_file):
    """The README tells users to stop the viewer with Ctrl+C. A real child process is started normally (default signal
    disposition), proven to be serving, interrupted with SIGINT, and must exit 0 with a clean stderr and a closed port."""
    import signal
    import threading
    import urllib.error
    import urllib.request

    proc = subprocess.Popen([sys.executable, "-m", "retrieval_eval.viewer", str(run_file), "--port", "0"],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    watchdog = threading.Timer(60, proc.kill)  # a hang becomes a failure, not a blocked test run
    watchdog.start()
    try:
        line = proc.stdout.readline()
        assert "Ctrl-C to stop" in line and "http://127.0.0.1:" in line
        port = int(line.split("127.0.0.1:")[1].split("/")[0])
        assert urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=10).status == 200
        proc.send_signal(signal.SIGINT)
        assert proc.wait(timeout=30) == 0
        assert proc.stderr.read() == ""
        with pytest.raises(urllib.error.URLError):
            urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=3)
    finally:
        watchdog.cancel()
        if proc.poll() is None:
            proc.kill()
        proc.stdout.close()
        proc.stderr.close()


def test_cli_serves_until_interrupted(run_file, monkeypatch, capsys):
    class Stub:
        server_port = 4321
        closed = False

        def serve_forever(self):
            raise KeyboardInterrupt

        def server_close(self):
            Stub.closed = True

    monkeypatch.setattr(viewer, "make_server", lambda run, port: Stub())
    assert viewer.main([str(run_file), "--port", "0"]) == 0 and Stub.closed
    out = capsys.readouterr().out
    assert "http://127.0.0.1:4321/" in out and "synthetic" in out


def test_cli_flushes_the_startup_url_before_serving(run_file, monkeypatch):
    import io

    raw = io.BytesIO()
    with io.TextIOWrapper(raw, encoding="utf-8") as stdout, monkeypatch.context() as patch:
        class Server:
            server_port = 4321

            def serve_forever(self):
                assert b"http://127.0.0.1:4321/" in raw.getvalue()
                raise KeyboardInterrupt

            def server_close(self):
                pass

        patch.setattr(sys, "stdout", stdout)
        patch.setattr(viewer, "make_server", lambda run, port: Server())
        assert viewer.main([str(run_file), "--port", "0"]) == 0


# ------------------------------------------------------------------ narrow-screen layout (structure; see evidence/ for a browser check)
class _TableScopes(html.parser.HTMLParser):
    """Records, for every <table>, whether it sits inside a div.table-scroll."""

    def __init__(self):
        super().__init__()
        self.stack, self.tables = [], []

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.tables.append(any(t == "div" and "table-scroll" in c.split() for t, c in self.stack))
        if tag in ("div", "table", "p", "ul", "form", "span"):
            self.stack.append((tag, dict(attrs).get("class", "") or ""))

    def handle_endtag(self, tag):
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i][0] == tag:
                del self.stack[i:]
                break


def test_every_table_scrolls_inside_its_own_region_and_the_page_declares_a_viewport(run):
    for qid in ("q01", "q02", "q08"):
        page = viewer.render_page(run, qid, 5)
        parser = _TableScopes()
        parser.feed(page)
        assert len(parser.tables) == 4 and all(parser.tables)  # quality, latency and one ranking table per method
        assert '<meta name="viewport" content="width=device-width, initial-scale=1">' in page
    assert page.count('class="table-scroll" role="region"') == 4 and page.count('tabindex="0"') == 4  # keyboard-scrollable
    assert ".table-scroll{max-width:100%;overflow-x:auto}" in viewer.STYLE and "select{max-width:100%}" in viewer.STYLE
    assert "grid-template-columns:1fr}" in viewer.STYLE and "@media (max-width:60rem)" in viewer.STYLE  # columns stack


def test_error_page_also_uses_the_scroll_regions(run):
    parser = _TableScopes()
    parser.feed(viewer.render_page(run, error="bad input"))
    assert parser.tables and all(parser.tables)


# ------------------------------------------------------------------ standalone: no retrieval/model code, no dataset, no cache
STANDALONE = r"""
import http.client, json, sys
BLOCKED = ["torch", "sentence_transformers", "transformers", "rank_bm25", "numpy", "huggingface_hub", "retrieval_eval.retrievers"]
for name in BLOCKED:
    sys.modules[name] = None            # any import of these now raises ImportError
from retrieval_eval import viewer
run = viewer.load_for_view(sys.argv[1])
rc = viewer.main([sys.argv[1], "--export-html", sys.argv[2], "--query", "q02", "--k", "5"])
srv, base = viewer.serve_in_thread(run)
conn = http.client.HTTPConnection("127.0.0.1", srv.server_port, timeout=10)
conn.request("GET", "/?q=q03&k=3"); r = conn.getresponse(); body = r.read().decode("utf-8")
srv.shutdown()
leaked = sorted(m for m in BLOCKED if sys.modules.get(m) is not None)
print(json.dumps({"rc": rc, "status": r.status, "has_q03": 'value="q03" selected' in body, "leaked": leaked}))
"""


def test_viewer_runs_with_retrieval_and_model_code_unavailable(tmp_path, run_file):
    elsewhere = tmp_path / "somewhere_else"  # no dataset directory, no model cache, different working directory
    elsewhere.mkdir()
    out = elsewhere / "page.html"
    env = {**os.environ, "HF_HOME": str(elsewhere / "empty_hf_home"), "HOME": str(elsewhere)}  # keep OS variables (Windows needs SYSTEMROOT)
    proc = subprocess.run([sys.executable, "-c", STANDALONE, str(run_file), str(out)], capture_output=True, text=True,
                          cwd=elsewhere, env=env, timeout=120)
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout.strip().splitlines()[-1])  # the last line; --export-html prints a 'wrote ...' line first
    assert result == {"rc": 0, "status": 200, "has_q03": True, "leaked": []}
    assert "d10: missed: not in the saved top-5" in out.read_text(encoding="utf-8")
    assert not (elsewhere / "empty_hf_home").exists()  # nothing tried to create a model cache


def test_viewer_module_imports_no_retrieval_code():
    src = Path(viewer.__file__).read_text(encoding="utf-8")
    for needle in ("retrievers", "rank_bm25", "sentence_transformers", "torch", "numpy", "huggingface_hub"):
        assert f"import {needle}" not in src and f"from .{needle}" not in src and f"from . import {needle}" not in src
