"""Unit tests for the retrieval adapters. No model, no network.

``FakeEncoder`` below is a CONTROLLED FAKE (a hand-written text->vector table). It exercises the
embedding adapter's normalization, exhaustive search, ordering and error handling only. It says nothing
about the real model; real-model behavior is tested in tests/test_retrievers_model.py (marker: model).
"""

import math
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from retrieval_eval.data import DataValidationError, Document, load_dataset
from retrieval_eval.metrics import RunContractError
from retrieval_eval.retrievers import (
    BM25_B,
    BM25_EPSILON,
    BM25_K1,
    TOKEN_PATTERN,
    BM25Retriever,
    EmbeddingRetriever,
    normalize_rows,
    tokenize,
)

FIXTURE = Path(__file__).resolve().parents[1] / "data" / "synthetic"


def docs(*pairs):
    return [Document(i, t) for i, t in pairs]


# ============================== tokenizer ======================================================
def test_tokenizer_contract():
    assert tokenize("E4107") == ["e4107"]  # identifier survives as ONE token
    assert tokenize("Error-E4107, ledger!") == ["error", "e4107", "ledger"]
    assert tokenize("SEC-2291") == ["sec", "2291"]
    assert tokenize("!!! ... ???") == []
    assert TOKEN_PATTERN == r"[a-z0-9]+"


def test_tokenizer_documented_ascii_only_limitation():
    # Intentional MVP limitation: non-ASCII letters are separators, no stemming.
    assert tokenize("café") == ["caf"]
    assert tokenize("pythons") == ["pythons"]  # no stemming


# ============================== BM25: hand-checkable case ======================================
BM25_DOCS = docs(
    ("d1", "Error E4107 ledger"),
    ("d2", "Error ledger timeout"),
    ("d3", "Password reset"),
)


def test_bm25_exact_identifier_hand_calculation():
    r = BM25Retriever(BM25_DOCS)
    results = r.search("E4107", 3)
    assert [d for d, _ in results] == ["d1", "d2", "d3"]
    # Only d1 contains token "e4107". Hand calculation of BM25Okapi (k1=1.5, b=0.75):
    #   N=3, n(e4107)=1  -> idf = ln(N - n + 0.5) - ln(n + 0.5) = ln(2.5) - ln(1.5)
    #   doc lengths 3,3,2 -> avgdl = 8/3 ; for d1: f=1, dl=3
    #   score = idf * f*(k1+1) / (f + k1*(1 - b + b*dl/avgdl))
    k1, b = 1.5, 0.75
    idf = math.log(3 - 1 + 0.5) - math.log(1 + 0.5)
    expected = idf * (1 * (k1 + 1)) / (1 + k1 * (1 - b + b * 3 / (8 / 3)))
    assert results[0][1] == pytest.approx(expected, rel=1e-12)
    assert results[0][1] == pytest.approx(0.4836, abs=5e-5)  # sanity value worked out by hand
    # Non-matching documents score exactly 0 and are ordered by ascending ID.
    assert results[1][1] == 0.0 and results[2][1] == 0.0


def test_bm25_is_case_insensitive_and_punctuation_insensitive():
    r = BM25Retriever(BM25_DOCS)
    assert r.search("e4107", 3) == r.search("E4107!!", 3) == r.search("  E4107  ", 3)


def test_bm25_all_zero_lexical_scores_give_deterministic_id_order():
    r = BM25Retriever(BM25_DOCS)
    for q in ["zzzz", "!!! ???"]:  # no matching term / no tokens at all (not blank text)
        res = r.search(q, 3)
        assert [d for d, _ in res] == ["d1", "d2", "d3"]
        assert [s for _, s in res] == [0.0, 0.0, 0.0]
        assert res == r.search(q, 3)  # repeatable


def test_bm25_ties_across_cutoff_use_ascending_id_regardless_of_input_order():
    # 5 documents so the query terms are rare enough (n=2 of N=5) to have positive idf.
    base = [("b", "same words here"), ("a", "same words here"), ("c", "other"),
            ("d", "unrelated stuff"), ("e", "more text")]
    for seed in range(6):
        shuffled = base[:]
        random.Random(seed).shuffle(shuffled)
        r = BM25Retriever(docs(*shuffled))
        assert [d for d, _ in r.search("same words", 1)] == ["a"]  # tie a/b cut at k=1
        assert [d for d, _ in r.search("same words", 2)] == ["a", "b"]


def test_bm25_topk_is_prefix_of_full_ranking_and_length_is_min_k_n():
    r = BM25Retriever(BM25_DOCS)
    full = r.search("error ledger", 3)
    assert r.search("error ledger", 2) == full[:2]
    assert r.search("error ledger", 1) == full[:1]
    assert len(r.search("error ledger", 50)) == 3  # k > corpus: no padding
    assert len(full) == 3


@pytest.mark.parametrize("bad_k", [0, -1, 1.0, 2.5, True, "3", None])
def test_search_rejects_invalid_k(bad_k):
    with pytest.raises(ValueError):
        BM25Retriever(BM25_DOCS).search("error", bad_k)


@pytest.mark.parametrize("bad_q", ["", "   ", "\n\t", None, 5, b"error"])
def test_search_rejects_blank_or_nonstring_query(bad_q):
    with pytest.raises(DataValidationError):
        BM25Retriever(BM25_DOCS).search(bad_q, 1)


def test_bm25_scores_are_plain_finite_python_floats_and_ids_are_strings():
    r = BM25Retriever(BM25_DOCS)
    for d, s in r.search("error ledger password", 3):
        assert type(d) is str and type(s) is float and math.isfinite(s)


def test_bm25_negative_scores_are_kept_not_dropped():
    # Terms in more than half the documents get a negative idf floor; with a tiny corpus the
    # epsilon * average_idf replacement can be negative. Results must still contain every doc.
    r = BM25Retriever(docs(("a", "x y"), ("b", "x y"), ("c", "x z")))
    res = r.search("x", 3)
    assert len(res) == 3 and all(math.isfinite(s) for _, s in res)


def test_bm25_library_floors_negative_idf_so_matching_docs_can_rank_below_nonmatching():
    # Observed rank-bm25 BM25Okapi behavior on a tiny corpus (NOT altered by the adapter): a query term
    # present in more than half the documents gets idf = epsilon * average_idf, here negative, so the
    # two matching documents score below the non-matching one. Pinned so any change is noticed.
    r = BM25Retriever(docs(("a", "same words here"), ("b", "same words here"), ("c", "other")))
    res = r.search("same words", 3)
    assert [d for d, _ in res] == ["c", "a", "b"]
    assert res[0][1] == 0.0 and res[1][1] < 0 and res[1][1] == res[2][1]


def test_bm25_records_explicit_parameters():
    info = BM25Retriever(BM25_DOCS).describe()
    assert (info["k1"], info["b"], info["epsilon"]) == (1.5, 0.75, 0.25) == (BM25_K1, BM25_B, BM25_EPSILON)
    assert info["tokenizer_pattern"] == r"[a-z0-9]+"
    assert info["variant"] == "BM25Okapi" and info["stemming"] is False and info["stop_words"] is None
    assert info["library_version"]  # recorded, nonempty


# ============================== construction validation ======================================
def test_construction_rejects_bad_corpora():
    with pytest.raises(ValueError, match="empty corpus"):
        BM25Retriever([])
    with pytest.raises(DataValidationError, match="duplicate"):
        BM25Retriever(docs(("a", "x"), ("a", "y")))
    with pytest.raises(DataValidationError, match="nonblank"):
        BM25Retriever(docs(("a", "  ")))
    with pytest.raises(DataValidationError, match="nonempty"):
        BM25Retriever(docs((" ", "x")))
    with pytest.raises(TypeError):
        BM25Retriever([("a", "x")])  # type: ignore[list-item]


def test_bm25_all_documents_tokenless_is_rejected_but_one_tokenless_doc_is_fine():
    with pytest.raises(DataValidationError, match="no document produces any lexical token"):
        BM25Retriever(docs(("a", "!!!"), ("b", "???")))
    r = BM25Retriever(docs(("a", "!!!"), ("b", "hello world"), ("c", "other text")))
    res = r.search("hello", 3)  # n=1 of N=3 -> positive idf; tokenless doc "a" scores 0 and is kept
    assert [d for d, _ in res] == ["b", "a", "c"] and res[0][1] > 0


def test_adapters_index_raw_text_unchanged_and_never_see_judgments():
    seen = {}

    class RecordingEncoder:
        def encode(self, texts):
            seen.setdefault("calls", []).append(list(texts))
            return np.ones((len(texts), 2))

        def describe(self):
            return {"model_id": "FAKE-recording"}

    corpus = docs(("d1", "Raw Text, with CASE & punctuation!"), ("d2", "second"))
    EmbeddingRetriever(corpus, RecordingEncoder()).search("Query As Typed?", 1)
    assert seen["calls"][0] == ["Raw Text, with CASE & punctuation!", "second"]  # no title/label added
    assert seen["calls"][1] == ["Query As Typed?"]  # raw query text, unmodified
    # Neither adapter's constructor or search accepts judgments.
    import inspect

    for cls in (BM25Retriever, EmbeddingRetriever):
        assert "qrels" not in inspect.signature(cls.__init__).parameters
        assert "qrels" not in inspect.signature(cls.search).parameters


def test_both_adapters_index_identical_ids():
    ds = load_dataset(FIXTURE)
    bm25 = BM25Retriever(ds.documents)
    emb = EmbeddingRetriever(ds.documents, FakeEncoder.for_texts([d.text for d in ds.documents]))
    assert bm25.doc_ids == emb.doc_ids == ds.doc_ids


# ============================== BM25 on the frozen fixture (structure only) ==================
def test_bm25_on_fixture_returns_valid_unique_known_ids_and_is_repeatable():
    ds = load_dataset(FIXTURE)
    r = BM25Retriever(ds.documents)
    known = set(ds.doc_ids)
    for q in ds.queries:
        full = r.search(q.text, len(ds.documents))
        ids = [d for d, _ in full]
        assert len(ids) == len(set(ids)) == 20 and set(ids) == known
        assert all(type(s) is float and math.isfinite(s) for _, s in full)
        assert full == r.search(q.text, 20)
        assert r.search(q.text, 5) == full[:5]
    with pytest.raises(ValueError):
        r.search("x", 0)


def test_bm25_fixture_identifier_query_matches_only_the_identifier_document():
    # Structural check, not a quality claim: exactly one fixture document contains the token e4107.
    ds = load_dataset(FIXTURE)
    holders = [d.doc_id for d in ds.documents if "e4107" in tokenize(d.text)]
    assert holders == ["d01"]
    top = BM25Retriever(ds.documents).search("E4107", 3)
    assert top[0][0] == "d01" and top[0][1] > 0 and top[1][1] == 0.0


# ============================== embedding adapter with a FAKE encoder ========================
class FakeEncoder:
    """CONTROLLED FAKE. Maps exact text -> a hand-chosen vector. Not a model."""

    def __init__(self, table, dtype=np.float32):
        self.table, self.dtype, self.calls = table, dtype, []

    @classmethod
    def for_texts(cls, texts):
        rng = np.random.default_rng(0)
        return cls({t: rng.normal(size=8) for t in texts} | {"E4107": rng.normal(size=8)})

    def encode(self, texts):
        self.calls.append(list(texts))
        return np.array([self.table[t] for t in texts], dtype=self.dtype)

    def describe(self):
        return {"model_id": "FAKE-table-encoder"}


EMB_DOCS = docs(("a", "doc a"), ("b", "doc b"), ("c", "doc c"))
EMB_TABLE = {
    "doc a": [5.0, 0.0, 0.0],  # deliberately unnormalized, differing scales
    "doc b": [0.0, 0.2, 0.0],
    "doc c": [3.0, 3.0, 0.0],
    "q": [2.0, 0.0, 0.0],
    "q ortho": [0.0, 0.0, 9.0],
}


def test_embedding_normalization_and_cosine_values_by_hand():
    r = EmbeddingRetriever(EMB_DOCS, FakeEncoder(EMB_TABLE))
    res = r.search("q", 3)
    # cos(q,a)=1 ; cos(q,c)=1/sqrt(2) ; cos(q,b)=0   (independent of vector magnitudes)
    assert [d for d, _ in res] == ["a", "c", "b"]
    assert res[0][1] == pytest.approx(1.0, abs=1e-12)
    assert res[1][1] == pytest.approx(1 / math.sqrt(2), abs=1e-12)
    assert res[2][1] == pytest.approx(0.0, abs=1e-12)


def test_embedding_scores_invariant_to_vector_scale():
    scaled = {k: [x * (7.3 if k.startswith("doc") else 0.01) for x in v] for k, v in EMB_TABLE.items()}
    a = EmbeddingRetriever(EMB_DOCS, FakeEncoder(EMB_TABLE)).search("q", 3)
    b = EmbeddingRetriever(EMB_DOCS, FakeEncoder(scaled)).search("q", 3)
    assert [d for d, _ in a] == [d for d, _ in b]
    assert [s for _, s in a] == pytest.approx([s for _, s in b], abs=1e-12)


def test_normalize_rows_gives_unit_norms():
    out = normalize_rows([[3.0, 4.0], [0.0, 2.0]], what="t")
    assert np.linalg.norm(out, axis=1) == pytest.approx([1.0, 1.0], abs=1e-15)


def test_embedding_scores_are_plain_floats_even_for_float32_encoder():
    for s_dtype in (np.float32, np.float64):
        res = EmbeddingRetriever(EMB_DOCS, FakeEncoder(EMB_TABLE, dtype=s_dtype)).search("q", 3)
        assert all(type(d) is str and type(s) is float and math.isfinite(s) for d, s in res)


def test_embedding_query_encoded_on_every_search_no_cache_and_docs_encoded_once():
    enc = FakeEncoder(EMB_TABLE)
    r = EmbeddingRetriever(EMB_DOCS, enc)
    assert len(enc.calls) == 1 and enc.calls[0] == ["doc a", "doc b", "doc c"]
    r.search("q", 2)
    r.search("q", 2)
    r.search("q ortho", 1)
    assert enc.calls[1:] == [["q"], ["q"], ["q ortho"]]  # three searches -> three query encodes


def test_embedding_ties_across_cutoff_by_ascending_id():
    table = {"x": [1.0, 0.0], "y": [1.0, 0.0], "z": [0.0, 1.0], "q": [1.0, 0.0]}
    for order in (["x", "y", "z"], ["y", "x", "z"], ["z", "y", "x"]):
        r = EmbeddingRetriever([Document(i, i) for i in order], FakeEncoder(table))
        assert [d for d, _ in r.search("q", 1)] == ["x"]
        assert [d for d, _ in r.search("q", 2)] == ["x", "y"]


def test_embedding_all_equal_scores_deterministic_order():
    # Query orthogonal to every document: all scores are (about) 0; result must still be ordered by ID.
    table = {"doc a": [1, 0, 0], "doc b": [1, 0, 0], "doc c": [1, 0, 0], "q": [0, 1, 0]}
    res = EmbeddingRetriever(EMB_DOCS, FakeEncoder(table)).search("q", 3)
    assert [d for d, _ in res] == ["a", "b", "c"] and [s for _, s in res] == [0.0, 0.0, 0.0]


def test_embedding_length_prefix_and_k_validation():
    r = EmbeddingRetriever(EMB_DOCS, FakeEncoder(EMB_TABLE))
    full = r.search("q", 3)
    assert r.search("q", 2) == full[:2] and len(r.search("q", 99)) == 3
    for bad in (0, -2, True, 1.5, "1"):
        with pytest.raises(ValueError):
            r.search("q", bad)
    with pytest.raises(DataValidationError):
        r.search("  ", 1)


@pytest.mark.parametrize(
    "bad_vec,match",
    [
        ([float("nan"), 1.0, 0.0], "NaN or infinite"),
        ([float("inf"), 1.0, 0.0], "NaN or infinite"),
        ([0.0, 0.0, 0.0], "zero norm"),
    ],
)
def test_bad_document_embeddings_are_rejected_not_repaired(bad_vec, match):
    table = dict(EMB_TABLE, **{"doc b": bad_vec})
    with pytest.raises(RunContractError, match=match):
        EmbeddingRetriever(EMB_DOCS, FakeEncoder(table))


def test_bad_query_embeddings_are_rejected_at_search():
    r = EmbeddingRetriever(EMB_DOCS, FakeEncoder(dict(EMB_TABLE, qnan=[float("nan"), 0, 0], qzero=[0, 0, 0], qdim=[1, 0])))
    with pytest.raises(RunContractError, match="NaN or infinite"):
        r.search("qnan", 1)
    with pytest.raises(RunContractError, match="zero norm"):
        r.search("qzero", 1)
    with pytest.raises(RunContractError, match="does not match document dimension"):
        r.search("qdim", 1)


def test_wrong_encoder_output_shapes_rejected():
    class Flat:
        def encode(self, texts):
            return np.ones(len(texts))  # 1-D

        def describe(self):
            return {}

    class WrongRows:
        def encode(self, texts):
            return np.ones((len(texts) + 1, 3))

        def describe(self):
            return {}

    with pytest.raises(RunContractError):
        EmbeddingRetriever(EMB_DOCS, Flat())
    with pytest.raises(RunContractError, match="rows"):
        EmbeddingRetriever(EMB_DOCS, WrongRows())


def test_encoder_errors_propagate_and_are_never_replaced_by_a_fallback():
    class Exploding:
        def encode(self, texts):
            raise RuntimeError("boom: model unavailable")

        def describe(self):
            return {}

    with pytest.raises(RuntimeError, match="boom"):  # at document-encoding time
        EmbeddingRetriever(EMB_DOCS, Exploding())

    class FailsOnQuery(FakeEncoder):
        def encode(self, texts):
            if texts == ["q"]:
                raise ConnectionError("network down")
            return super().encode(texts)

    r = EmbeddingRetriever(EMB_DOCS, FailsOnQuery(EMB_TABLE))
    with pytest.raises(ConnectionError, match="network down"):  # at query-encoding time
        r.search("q", 1)


def test_embedding_describe_reports_config():
    info = EmbeddingRetriever(EMB_DOCS, FakeEncoder(EMB_TABLE)).describe()
    assert info["query_cache"] is False and info["dimension"] == 3 and info["n_documents"] == 3
    assert info["encoder"] == {"model_id": "FAKE-table-encoder"}
    assert "L2" in info["normalization"]


# ============================== import safety ==================================================
def test_importing_adapters_does_not_load_torch_or_sentence_transformers():
    code = (
        "import sys, retrieval_eval.retrievers; "
        "bad = [m for m in ('torch', 'sentence_transformers', 'transformers') if m in sys.modules]; "
        "sys.exit(1 if bad else 0)"
    )
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
