import json
import shutil
from pathlib import Path

import pytest

from retrieval_eval.data import DataValidationError, load_dataset, load_documents, load_qrels, load_queries

FIXTURE = Path(__file__).resolve().parents[1] / "data" / "synthetic"

# Frozen judgments (authored before any retriever was run). Changing the fixture's
# positives must be a deliberate edit to BOTH the fixture and this table.
EXPECTED_POSITIVES = {
    "q01": {"d01"},
    "q02": {"d10"},
    "q03": {"d02", "d03"},
    "q04": {"d07"},
    "q05": {"d05"},
    "q06": {"d14"},
    "q07": {"d18", "d19"},
    "q08": set(),
}


# --- the real fixture --------------------------------------------------------------------
def test_fixture_loads_with_expected_shape():
    ds = load_dataset(FIXTURE)
    assert len(ds.documents) == 20
    assert len(ds.queries) == 8
    assert sum(len(v) for v in ds.qrels.values()) == 160


def test_fixture_judgments_are_complete_and_binary():
    ds = load_dataset(FIXTURE)
    assert ds.missing_judgments() == []
    assert {r for per_q in ds.qrels.values() for r in per_q.values()} == {0, 1}


def test_fixture_frozen_positives():
    ds = load_dataset(FIXTURE)
    assert {q: set(ds.relevant_ids(q)) for q in ds.query_ids} == EXPECTED_POSITIVES


def test_fixture_covers_required_scenarios():
    ds = load_dataset(FIXTURE)
    sizes = [len(ds.relevant_ids(q)) for q in ds.query_ids]
    assert 0 in sizes, "needs a no-relevant query"
    assert max(sizes) >= 2, "needs a multi-relevant query"
    # no-relevant query is present with explicit zero judgments for every document
    q_none = next(q for q in ds.query_ids if not ds.relevant_ids(q))
    assert len(ds.qrels[q_none]) == len(ds.documents)
    assert any(q.text == "E4107" for q in ds.queries)  # exact identifier query


def test_labels_distinguish_unjudged_from_nonrelevant(tmp_path):
    d = _copy_fixture(tmp_path)
    rows = (d / "qrels.csv").read_text().splitlines()
    rows = [r for r in rows if r != "q01,d02,0"]  # drop one judgment
    (d / "qrels.csv").write_text("\n".join(rows) + "\n")
    ds = load_dataset(d)
    assert ds.label("q01", "d02") == "unjudged"
    assert ds.label("q01", "d03") == "nonrelevant"
    assert ds.label("q01", "d01") == "relevant"
    assert ("q01", "d02") in ds.missing_judgments()


def test_fixture_texts_are_short_enough_for_model_limit():
    # model default truncation is 256 word pieces; ~200 chars is far below that
    ds = load_dataset(FIXTURE)
    assert max(len(d.text) for d in ds.documents) < 200


# --- invalid inputs ----------------------------------------------------------------------
def _copy_fixture(tmp_path) -> Path:
    dst = tmp_path / "ds"
    shutil.copytree(FIXTURE, dst)
    return dst


def _write_lines(path, lines):
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_missing_file(tmp_path):
    with pytest.raises(DataValidationError, match="file not found"):
        load_dataset(tmp_path)


@pytest.mark.parametrize(
    "line,match",
    [
        ('{"doc_id": "a", "text": "x"}\n{"doc_id": "a", "text": "y"}', "duplicate doc_id 'a'"),
        ('{"doc_id": "a", "text": "   "}', "text must not be blank"),
        ('{"doc_id": "a"}', "missing field"),
        ('{"doc_id": "a", "text": "x", "extra": 1}', "unexpected field"),
        ('{"doc_id": "a", "text": "x"\n', "invalid JSON"),
        ('{"doc_id": 7, "text": "x"}', "must be a string"),
        ('{"doc_id": "", "text": "x"}', "nonempty"),
        ('{"doc_id": "  ", "text": "x"}', "nonempty"),
        ('{"doc_id": "a", "text": 5}', "text must be a string"),
        ('["a", "x"]', "expected a JSON object"),
    ],
)
def test_invalid_documents(tmp_path, line, match):
    p = tmp_path / "documents.jsonl"
    p.write_text(line + "\n", encoding="utf-8")
    with pytest.raises(DataValidationError, match=match):
        load_documents(p)


def test_error_reports_line_number(tmp_path):
    p = tmp_path / "documents.jsonl"
    _write_lines(p, ['{"doc_id": "a", "text": "ok"}', '{"doc_id": "b", "text": ""}'])
    with pytest.raises(DataValidationError, match=r"documents\.jsonl:2"):
        load_documents(p)


def test_blank_corpus_and_query_set(tmp_path):
    for name, loader in [("documents.jsonl", load_documents), ("queries.jsonl", load_queries)]:
        p = tmp_path / name
        p.write_text("\n\n", encoding="utf-8")
        with pytest.raises(DataValidationError, match="empty"):
            loader(p)


def test_duplicate_query_id(tmp_path):
    p = tmp_path / "queries.jsonl"
    _write_lines(p, ['{"query_id": "q", "text": "a"}', '{"query_id": "q", "text": "b"}'])
    with pytest.raises(DataValidationError, match="duplicate query_id"):
        load_queries(p)


def test_ids_are_case_sensitive_and_preserved(tmp_path):
    p = tmp_path / "documents.jsonl"
    _write_lines(p, ['{"doc_id": "Doc", "text": "a"}', '{"doc_id": "doc", "text": "b"}'])
    docs = load_documents(p)
    assert [d.doc_id for d in docs] == ["Doc", "doc"]


def test_non_utf8_rejected(tmp_path):
    p = tmp_path / "documents.jsonl"
    p.write_bytes(b'{"doc_id": "a", "text": "caf\xe9"}\n')
    with pytest.raises(DataValidationError, match="UTF-8"):
        load_documents(p)


def _qrels(tmp_path, lines):
    p = tmp_path / "qrels.csv"
    _write_lines(p, lines)
    return p


def test_qrels_bad_header(tmp_path):
    with pytest.raises(DataValidationError, match="header"):
        load_qrels(_qrels(tmp_path, ["query,doc,rel", "q1,d1,1"]))
    p = tmp_path / "empty.csv"
    p.write_text("", encoding="utf-8")
    with pytest.raises(DataValidationError, match="header"):
        load_qrels(p)


@pytest.mark.parametrize("bad", ["2", "-1", "1.0", "true", "", " 1", "01", "0.0"])
def test_qrels_rejects_non_binary_relevance(tmp_path, bad):
    with pytest.raises(DataValidationError, match="relevance must be exactly 0 or 1"):
        load_qrels(_qrels(tmp_path, ["query_id,doc_id,relevance", f"q1,d1,{bad}"]))


def test_qrels_duplicate_pair_rejected_even_if_identical(tmp_path):
    with pytest.raises(DataValidationError, match="duplicate qrel pair"):
        load_qrels(_qrels(tmp_path, ["query_id,doc_id,relevance", "q1,d1,1", "q1,d1,1"]))
    with pytest.raises(DataValidationError, match="duplicate qrel pair"):
        load_qrels(_qrels(tmp_path, ["query_id,doc_id,relevance", "q1,d1,1", "q1,d1,0"]))


def test_qrels_wrong_column_count_and_empty_ids(tmp_path):
    with pytest.raises(DataValidationError, match="3 columns"):
        load_qrels(_qrels(tmp_path, ["query_id,doc_id,relevance", "q1,d1"]))
    with pytest.raises(DataValidationError, match="3 columns"):
        load_qrels(_qrels(tmp_path, ["query_id,doc_id,relevance", "q1,d1,1,x"]))
    with pytest.raises(DataValidationError, match="nonempty"):
        load_qrels(_qrels(tmp_path, ["query_id,doc_id,relevance", ",d1,1"]))


def test_unknown_ids_in_qrels_rejected(tmp_path):
    d = _copy_fixture(tmp_path)
    with open(d / "qrels.csv", "a", encoding="utf-8") as f:
        f.write("q01,d99,0\n")
    with pytest.raises(DataValidationError, match="unknown doc_id.*d99"):
        load_dataset(d)
    d2 = tmp_path / "ds2"
    shutil.copytree(FIXTURE, d2)
    with open(d2 / "qrels.csv", "a", encoding="utf-8") as f:
        f.write("q99,d01,0\n")
    with pytest.raises(DataValidationError, match="unknown query_id.*q99"):
        load_dataset(d2)


def test_query_without_qrel_rows_rejected(tmp_path):
    d = _copy_fixture(tmp_path)
    rows = [r for r in (d / "qrels.csv").read_text().splitlines() if not r.startswith("q08,")]
    (d / "qrels.csv").write_text("\n".join(rows) + "\n")
    with pytest.raises(DataValidationError, match="no qrel rows.*q08"):
        load_dataset(d)


def test_fixture_json_is_stable_utf8_without_bom():
    for name in ("documents.jsonl", "queries.jsonl", "qrels.csv"):
        raw = (FIXTURE / name).read_bytes()
        assert not raw.startswith(b"\xef\xbb\xbf")
        raw.decode("utf-8")
    for line in (FIXTURE / "documents.jsonl").read_text(encoding="utf-8").splitlines():
        json.loads(line)


# === regression tests for handoff-02 review findings =======================================
# Finding 4: repeated keys inside a JSON object must be rejected, not silently overwritten.
@pytest.mark.parametrize(
    "line,key",
    [
        ('{"doc_id":"original","doc_id":"overwritten","text":"x"}', "doc_id"),
        ('{"doc_id":"a","text":"first","text":"second"}', "text"),
        ('{"doc_id":"a","text":"same","text":"same"}', "text"),  # identical repeat still rejected
        ('{"doc_id":"a","text":"x","extra":{"k":1,"k":2}}', "k"),  # nested repeat
    ],
)
def test_repeated_json_keys_rejected_in_documents(tmp_path, line, key):
    p = tmp_path / "documents.jsonl"
    p.write_text(line + "\n", encoding="utf-8")
    with pytest.raises(DataValidationError, match=rf"documents\.jsonl:1: duplicate key '{key}'"):
        load_documents(p)


def test_repeated_json_keys_rejected_in_queries_with_line_number(tmp_path):
    p = tmp_path / "queries.jsonl"
    _write_lines(p, ['{"query_id":"q1","text":"ok"}',
                     '{"query_id":"q2","query_id":"q3","text":"x"}'])
    with pytest.raises(DataValidationError, match=r"queries\.jsonl:2: duplicate key 'query_id'"):
        load_queries(p)
    p.write_text('{"query_id":"q","text":"a","text":"b"}\n', encoding="utf-8")
    with pytest.raises(DataValidationError, match="duplicate key 'text'"):
        load_queries(p)


def test_distinct_keys_still_load_after_duplicate_key_hook(tmp_path):
    p = tmp_path / "documents.jsonl"
    _write_lines(p, ['{"doc_id":"a","text":"x"}', '{"text":"y","doc_id":"b"}'])
    assert [(d.doc_id, d.text) for d in load_documents(p)] == [("a", "x"), ("b", "y")]


def test_fixture_manifest_matches_files():
    """FIXTURE.sha256 is the authoritative manifest; the legacy SHA256SUMS is deliberately not checked."""
    import hashlib

    lines = (FIXTURE / "FIXTURE.sha256").read_text(encoding="utf-8").splitlines()
    entries = {}
    for line in lines:
        digest, name = line.split("  ", 1)
        entries[name] = digest
    assert set(entries) == {"documents.jsonl", "queries.jsonl", "qrels.csv"}
    for name, digest in entries.items():
        assert hashlib.sha256((FIXTURE / name).read_bytes()).hexdigest() == digest, name
