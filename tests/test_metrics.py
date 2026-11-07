import json
import math
import random
from fractions import Fraction

import pytest

from retrieval_eval.metrics import (
    RunContractError,
    evaluate_run,
    order_scored_results,
    recall_at_k,
    rr_at_k,
    validate_k,
)

# --- handoff section 4 oracle table: (relevant, ranking, k, recall, rr) -----------------
ORACLE = [
    ({"A", "C"}, ["B", "C", "A"], 1, 0.0, 0.0),
    ({"A", "C"}, ["B", "C", "A"], 3, 1.0, 0.5),
    ({"A", "C"}, ["C"], 3, 0.5, 1.0),
    ({"A", "C"}, [], 3, 0.0, 0.0),
    ({"A"}, ["B", "C", "A"], 2, 0.0, 0.0),
    ({"A", "B", "C", "D"}, ["A", "B"], 2, 0.5, 1.0),
    (set(), ["A"], 3, None, None),
]


@pytest.mark.parametrize("relevant,ranking,k,recall,rr", ORACLE)
def test_oracle_recall_and_rr(relevant, ranking, k, recall, rr):
    assert recall_at_k(relevant, ranking, k) == recall
    assert rr_at_k(relevant, ranking, k) == rr


def test_k_larger_than_ranking_does_not_pad_and_keeps_full_denominator():
    # k=10 > corpus/ranking size: use available results; denominator stays |R|=4.
    assert recall_at_k({"A", "B", "C", "D"}, ["A", "B"], 10) == 0.5


@pytest.mark.parametrize("bad", [0, -1, 1.0, 2.5, True, False, "3", None])
def test_invalid_k_rejected(bad):
    with pytest.raises(ValueError):
        validate_k(bad)
    with pytest.raises(ValueError):
        recall_at_k({"A"}, ["A"], bad)
    with pytest.raises(ValueError):
        rr_at_k({"A"}, ["A"], bad)


def test_duplicate_in_ranking_rejected_even_beyond_k():
    # duplicate sits past the cutoff; must still be rejected before truncation
    with pytest.raises(RunContractError, match="duplicate"):
        recall_at_k({"A"}, ["A", "B", "C", "B"], 1)
    with pytest.raises(RunContractError, match="duplicate"):
        rr_at_k({"A"}, ["A", "B", "C", "B"], 1)


# --- macro oracle ------------------------------------------------------------------------
CORPUS = ["A", "B", "C", "D"]
QRELS = {
    "q1": {"A": 1, "B": 0, "C": 1, "D": 0},
    "q2": {"A": 0, "B": 0, "C": 0, "D": 1},
    "q3": {"A": 0, "B": 0, "C": 0, "D": 0},
}


def _macro(rankings, qrels=QRELS, ks=(2,)):
    return evaluate_run(
        query_ids=list(qrels), qrels=qrels, rankings=rankings, corpus_ids=CORPUS, ks=ks
    )


def test_macro_oracle_denominators_and_nulls():
    run = _macro({"q1": ["B", "C"], "q2": [], "q3": ["A"]})
    q1, q2, q3 = run.per_query
    assert (q1.recall[2], q1.rr[2]) == (0.5, 0.5)
    assert (q2.recall[2], q2.rr[2]) == (0.0, 0.0)
    assert (q3.recall[2], q3.rr[2]) == (None, None)
    assert q3.status == "no_judged_relevant" and q1.status == "scored"
    assert run.aggregate[2].macro_recall == 0.25
    assert run.aggregate[2].mrr == 0.25
    assert (run.total_queries, run.scored_queries, run.no_relevant_queries) == (3, 2, 1)
    # no-relevant query stays represented
    assert [q.query_id for q in run.per_query] == ["q1", "q2", "q3"]


def test_macro_is_not_pooled_recall():
    # Pooled: (1 hit) / (2 + 1 relevant) = 1/3 ; macro must be 1/4.
    run = _macro({"q1": ["B", "C"], "q2": [], "q3": []})
    assert run.aggregate[2].macro_recall == 0.25
    assert run.aggregate[2].macro_recall != pytest.approx(1 / 3)


def test_empty_ranking_for_answerable_query_stays_in_denominator():
    run = _macro({"q1": [], "q2": [], "q3": []})
    assert run.aggregate[2].macro_recall == 0.0
    assert run.aggregate[2].mrr == 0.0
    assert run.scored_queries == 2


def test_all_queries_unscorable_gives_null_not_nan_or_zero():
    qrels = {"q3": {"A": 0, "B": 0, "C": 0, "D": 0}}
    run = _macro({"q3": ["A"]}, qrels=qrels)
    agg = run.aggregate[2]
    assert agg.macro_recall is None and agg.mrr is None
    assert run.scored_queries == 0 and run.no_relevant_queries == 1
    d = run.to_dict()
    assert d["aggregate"]["@2"] == {"macro_recall": None, "mrr": None}
    assert "NaN" not in json.dumps(d)  # JSON null, never NaN


def test_multiple_ks_derived_from_one_ranking():
    run = _macro({"q1": ["B", "C", "A"], "q2": ["D"], "q3": []}, ks=(1, 3))
    q1 = run.per_query[0]
    assert (q1.recall[1], q1.rr[1]) == (0.0, 0.0)
    assert (q1.recall[3], q1.rr[3]) == (1.0, 0.5)
    assert run.aggregate[1].macro_recall == 0.5  # (0 + 1) / 2
    assert run.aggregate[3].mrr == 0.75  # (0.5 + 1) / 2


def test_ks_validation():
    for bad in ([], [0], [1, 1], [True], [1.5]):
        with pytest.raises(ValueError):
            _macro({"q1": [], "q2": [], "q3": []}, ks=bad)


# --- run-contract errors -------------------------------------------------------------------
def test_omitted_query_is_contract_error_but_explicit_empty_is_fine():
    with pytest.raises(RunContractError, match="no ranking record"):
        _macro({"q1": ["A"], "q2": []})  # q3 omitted
    _macro({"q1": ["A"], "q2": [], "q3": []})  # explicit empty list accepted


def test_unexpected_query_record_rejected():
    with pytest.raises(RunContractError, match="unexpected"):
        _macro({"q1": [], "q2": [], "q3": [], "q9": []})


def test_unknown_doc_id_in_ranking_rejected():
    with pytest.raises(RunContractError, match="unknown doc_id 'Z'"):
        _macro({"q1": ["Z"], "q2": [], "q3": []})


def test_duplicate_doc_id_in_ranking_rejected_in_run():
    with pytest.raises(RunContractError, match="duplicate"):
        _macro({"q1": ["A", "A"], "q2": [], "q3": []})


def test_valid_unjudged_document_in_ranking_counts_zero_and_keeps_rank():
    qrels = {"q1": {"A": 1}}  # B, C, D are valid corpus docs but unjudged for q1
    run = _macro({"q1": ["B", "A"]}, qrels=qrels)
    assert run.per_query[0].rr[2] == 0.5  # unjudged B occupies rank 1
    assert run.per_query[0].recall[2] == 1.0


def test_qrels_problems_rejected():
    with pytest.raises(ValueError, match="unknown query_id"):
        evaluate_run(query_ids=["q1"], qrels={"q1": {"A": 1}, "qX": {"A": 1}},
                     rankings={"q1": []}, corpus_ids=CORPUS, ks=[1])
    with pytest.raises(ValueError, match="unknown doc_id"):
        _macro({"q1": []}, qrels={"q1": {"Z": 1}})
    with pytest.raises(ValueError, match="must be 0 or 1"):
        _macro({"q1": []}, qrels={"q1": {"A": 2}})
    with pytest.raises(ValueError, match="must be 0 or 1"):
        _macro({"q1": []}, qrels={"q1": {"A": True}})
    with pytest.raises(ValueError, match="no qrel rows"):
        evaluate_run(query_ids=["q1", "q2"], qrels={"q1": {"A": 1}},
                     rankings={"q1": [], "q2": []}, corpus_ids=CORPUS, ks=[1])


def test_blank_corpus_or_query_set_rejected():
    with pytest.raises(ValueError, match="corpus is empty"):
        evaluate_run(query_ids=["q1"], qrels={"q1": {}}, rankings={"q1": []}, corpus_ids=[], ks=[1])
    with pytest.raises(ValueError, match="query set is empty"):
        evaluate_run(query_ids=[], qrels={}, rankings={}, corpus_ids=CORPUS, ks=[1])


# --- deterministic ordering and ties ---------------------------------------------------------
def test_tie_breaks_by_ascending_doc_id():
    ordered = order_scored_results([("B", 1.0), ("A", 1.0)])
    ranking = [d for d, _ in ordered]
    assert ranking == ["A", "B"]
    assert rr_at_k({"B"}, ranking, 1) == 0.0
    assert rr_at_k({"B"}, ranking, 2) == 0.5


def test_ordering_is_score_desc_then_id_and_case_sensitive():
    ordered = order_scored_results([("b", 0.5), ("B", 0.5), ("c", 0.9), ("a", 0.5)])
    assert [d for d, _ in ordered] == ["c", "B", "a", "b"]  # 'B' (0x42) < 'a' (0x61)
    assert order_scored_results([("x", 1), ("y", 2.0)]) == [("y", 2.0), ("x", 1.0)]


def test_ordering_independent_of_input_order():
    items = [("d3", 0.2), ("d1", 0.2), ("d2", 0.7), ("d0", 0.2)]
    expected = order_scored_results(items)
    for seed in range(5):
        shuffled = items[:]
        random.Random(seed).shuffle(shuffled)
        assert order_scored_results(shuffled) == expected


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_nonfinite_scores_rejected(bad):
    with pytest.raises(RunContractError, match="non-finite"):
        order_scored_results([("A", 1.0), ("B", bad)])


def test_bad_score_types_and_duplicate_ids_rejected():
    with pytest.raises(RunContractError):
        order_scored_results([("A", "0.5")])
    with pytest.raises(RunContractError):
        order_scored_results([("A", True)])
    with pytest.raises(RunContractError, match="duplicate"):
        order_scored_results([("A", 1.0), ("A", 2.0)])


# --- independent cross-check against a naive exact-Fraction implementation -------------------
def _naive(relevant, ranking, k):
    if not relevant:
        return None, None
    top = ranking[:k]
    recall = Fraction(sum(1 for d in top if d in relevant), len(relevant))
    rr = Fraction(0)
    for i, d in enumerate(top, 1):
        if d in relevant:
            rr = Fraction(1, i)
            break
    return recall, rr


def test_randomized_agreement_with_naive_fraction_oracle():
    rng = random.Random(20261106)
    universe = [f"d{i}" for i in range(12)]
    for _ in range(300):
        relevant = set(rng.sample(universe, rng.randint(0, 5)))
        ranking = rng.sample(universe, rng.randint(0, 12))
        k = rng.randint(1, 15)
        exp_recall, exp_rr = _naive(relevant, ranking, k)
        got_recall = recall_at_k(relevant, ranking, k)
        got_rr = rr_at_k(relevant, ranking, k)
        if exp_recall is None:
            assert got_recall is None and got_rr is None
        else:
            assert got_recall == pytest.approx(float(exp_recall), abs=1e-15)
            assert got_rr == pytest.approx(float(exp_rr), abs=1e-15)


# === regression tests for handoff-02 review findings (all via evaluate_run) ================
def _run_one(ranking, qrel=None, corpus=("A", "B"), ks=(1, 2)):
    qrel = {"A": 1, "B": 0} if qrel is None else qrel
    return evaluate_run(
        query_ids=["q"], qrels={"q": qrel}, rankings={"q": ranking},
        corpus_ids=list(corpus), ks=list(ks),
    )


# Finding 1: string/bytes rankings must be rejected BEFORE list() conversion.
def test_string_ranking_rejected_not_split_into_characters():
    with pytest.raises(RunContractError, match="not str"):
        _run_one("BA")  # reproducer: previously scored Recall@2=1, RR@2=0.5


def test_multichar_string_ranking_gets_container_error_not_misleading_unknown_id():
    with pytest.raises(RunContractError, match="not str") as exc:
        _run_one("d01", qrel={"d01": 1}, corpus=("d01",), ks=(1,))
    assert "unknown doc_id" not in str(exc.value)


@pytest.mark.parametrize(
    "bad", [b"BA", bytearray(b"BA"), {"A", "B"}, frozenset({"A"}), {"A": 1}, None, 7, iter(["A"])]
)
def test_non_sequence_rankings_rejected(bad):
    with pytest.raises(RunContractError, match="list or tuple"):
        _run_one(bad)


def test_valid_list_tuple_and_explicit_empty_rankings_preserved():
    by_list = _run_one(["B", "A"])
    by_tuple = _run_one(("B", "A"))
    assert by_list == by_tuple
    assert (by_list.per_query[0].recall[2], by_list.per_query[0].rr[2]) == (1.0, 0.5)
    empty = _run_one([])
    assert (empty.per_query[0].recall[2], empty.per_query[0].rr[2]) == (0.0, 0.0)
    assert empty.scored_queries == 1


def test_standalone_helpers_also_reject_string_and_set_rankings():
    for bad in ("BA", b"BA", {"A"}):
        with pytest.raises(RunContractError):
            recall_at_k({"A"}, bad, 1)
        with pytest.raises(RunContractError):
            rr_at_k({"A"}, bad, 1)


# Finding 2: an empty per-query judgment mapping is "no qrel rows", not "no relevant".
def test_empty_qrel_mapping_rejected():
    with pytest.raises(ValueError, match="no qrel rows"):
        evaluate_run(query_ids=["q"], qrels={"q": {}}, rankings={"q": ["A"]},
                     corpus_ids=["A"], ks=[1])


def test_empty_mapping_for_one_query_among_valid_queries_rejected():
    with pytest.raises(ValueError, match=r"no qrel rows.*q2"):
        evaluate_run(
            query_ids=["q1", "q2"], qrels={"q1": {"A": 1}, "q2": {}},
            rankings={"q1": ["A"], "q2": ["A"]}, corpus_ids=["A"], ks=[1],
        )


def test_deliberate_no_relevant_query_with_explicit_zero_remains_valid_and_unscored():
    run = evaluate_run(query_ids=["q"], qrels={"q": {"A": 0}}, rankings={"q": ["A"]},
                       corpus_ids=["A"], ks=[1])
    q = run.per_query[0]
    assert q.status == "no_judged_relevant" and (q.recall[1], q.rr[1]) == (None, None)
    assert (run.total_queries, run.scored_queries, run.no_relevant_queries) == (1, 0, 1)
    assert run.aggregate[1].macro_recall is None and run.aggregate[1].mrr is None


# Finding 3: relevance must be the integer 0 or 1 at the evaluation boundary.
@pytest.mark.parametrize("bad", [1.0, 0.0, 1.5, "1", "0", None, 2, -1, True, False])
def test_non_integer_or_out_of_range_relevance_rejected(bad):
    with pytest.raises(ValueError, match="must be 0 or 1"):
        _run_one(["A"], qrel={"A": bad, "B": 0}, ks=(1,))


def test_integer_zero_and_one_accepted():
    run = _run_one(["A"], qrel={"A": 1, "B": 0}, ks=(1,))
    assert run.per_query[0].recall[1] == 1.0
