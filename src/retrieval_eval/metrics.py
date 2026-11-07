"""Recall@k, RR@k, macro aggregation, and deterministic ranking order.

Definitions and edge-case policy follow handoff section 4:
  Recall@k(q) = |R(q) ∩ L_k(q)| / |R(q)|        RR@k(q) = 1/r for first relevant at rank r <= k else 0
  Macro Recall@k / MRR@k = arithmetic mean over queries with >=1 judged-relevant document.
Queries with no judged-relevant documents get null metrics, are excluded from macro means,
and are counted separately. No NaN is ever produced.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real

STATUS_SCORED = "scored"
STATUS_NO_RELEVANT = "no_judged_relevant"


class RunContractError(ValueError):
    """Raised when a run's rankings violate the contract with the expected inputs."""


def validate_k(k) -> int:
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError(f"k must be a positive integer, got {k!r}")
    return k


def validate_ks(ks: Sequence[int]) -> tuple[int, ...]:
    ks = tuple(ks)
    if not ks:
        raise ValueError("ks must not be empty")
    for k in ks:
        validate_k(k)
    if len(set(ks)) != len(ks):
        raise ValueError(f"ks contains duplicates: {ks}")
    return ks


def _check_ranking_container(ranking, where: str) -> None:
    """Reject anything that is not an ordered sequence of IDs. Must run on the RAW value,
    before any list() conversion: list("BA") would silently split a string into characters."""
    if isinstance(ranking, (str, bytes, bytearray)):
        raise RunContractError(
            f"{where}: ranking must be a list or tuple of doc IDs, not {type(ranking).__name__}"
        )
    if not isinstance(ranking, Sequence):
        raise RunContractError(
            f"{where}: ranking must be a list or tuple of doc IDs, got {type(ranking).__name__}"
        )


def _validate_ranking(ranking: Sequence[str], corpus_ids: Collection[str] | None, where: str):
    _check_ranking_container(ranking, where)
    seen: set[str] = set()
    for doc_id in ranking:
        if not isinstance(doc_id, str):
            raise RunContractError(f"{where}: ranked ID {doc_id!r} is not a string")
        if doc_id in seen:  # checked on the FULL ranking, before any truncation
            raise RunContractError(f"{where}: duplicate doc_id {doc_id!r} in ranking")
        seen.add(doc_id)
        if corpus_ids is not None and doc_id not in corpus_ids:
            raise RunContractError(f"{where}: unknown doc_id {doc_id!r} in ranking")


def recall_at_k(relevant: Collection[str], ranking: Sequence[str], k: int) -> float | None:
    """None when there is no judged-relevant document. Denominator is always |R(q)|."""
    k = validate_k(k)
    _validate_ranking(ranking, None, "ranking")
    rel = set(relevant)
    if not rel:
        return None
    return len(rel.intersection(ranking[:k])) / len(rel)


def rr_at_k(relevant: Collection[str], ranking: Sequence[str], k: int) -> float | None:
    k = validate_k(k)
    _validate_ranking(ranking, None, "ranking")
    rel = set(relevant)
    if not rel:
        return None
    for rank, doc_id in enumerate(ranking[:k], 1):
        if doc_id in rel:
            return 1.0 / rank
    return 0.0


def order_scored_results(scored: Sequence[tuple[str, float]]) -> list[tuple[str, float]]:
    """Total order: score descending, then doc_id ascending (case-sensitive code-point order).

    Rejects duplicate IDs and non-finite / non-numeric scores. Returns the full ordering;
    truncate to k afterwards.
    """
    seen: set[str] = set()
    clean: list[tuple[str, float]] = []
    for doc_id, score in scored:
        if not isinstance(doc_id, str):
            raise RunContractError(f"scored doc_id {doc_id!r} is not a string")
        if doc_id in seen:
            raise RunContractError(f"duplicate doc_id {doc_id!r} in scored results")
        seen.add(doc_id)
        if isinstance(score, bool) or not isinstance(score, Real):
            raise RunContractError(f"score for {doc_id!r} is not a real number: {score!r}")
        value = float(score)
        if not math.isfinite(value):
            raise RunContractError(f"non-finite score for {doc_id!r}: {score!r}")
        clean.append((doc_id, value))
    return sorted(clean, key=lambda item: (-item[1], item[0]))


@dataclass(frozen=True)
class QueryMetrics:
    query_id: str
    status: str
    n_relevant: int
    n_returned: int
    recall: dict[int, float | None]
    rr: dict[int, float | None]


@dataclass(frozen=True)
class AggregateMetrics:
    macro_recall: float | None
    mrr: float | None


@dataclass(frozen=True)
class RunMetrics:
    ks: tuple[int, ...]
    per_query: tuple[QueryMetrics, ...]
    aggregate: dict[int, AggregateMetrics]
    total_queries: int
    scored_queries: int
    no_relevant_queries: int

    def to_dict(self) -> dict:
        return {
            "ks": list(self.ks),
            "counts": {
                "total": self.total_queries,
                "scored": self.scored_queries,
                "no_judged_relevant": self.no_relevant_queries,
            },
            "per_query": [
                {
                    "query_id": q.query_id,
                    "status": q.status,
                    "n_relevant": q.n_relevant,
                    "n_returned": q.n_returned,
                    **{f"recall@{k}": q.recall[k] for k in self.ks},
                    **{f"rr@{k}": q.rr[k] for k in self.ks},
                }
                for q in self.per_query
            ],
            "aggregate": {
                f"@{k}": {"macro_recall": a.macro_recall, "mrr": a.mrr}
                for k, a in self.aggregate.items()
            },
        }


def evaluate_run(
    *,
    query_ids: Sequence[str],
    qrels: Mapping[str, Mapping[str, int]],
    rankings: Mapping[str, Sequence[str]],
    corpus_ids: Collection[str],
    ks: Sequence[int],
) -> RunMetrics:
    ks = validate_ks(ks)
    query_ids = list(query_ids)
    if not query_ids:
        raise ValueError("query set is empty")
    if len(set(query_ids)) != len(query_ids):
        raise ValueError("duplicate query IDs in query_ids")
    corpus = set(corpus_ids)
    if not corpus:
        raise ValueError("corpus is empty")
    expected = set(query_ids)

    for qid, per_q in qrels.items():
        if qid not in expected:
            raise ValueError(f"qrels reference unknown query_id {qid!r}")
        for did, rel in per_q.items():
            if did not in corpus:
                raise ValueError(f"qrels reference unknown doc_id {did!r} for query {qid!r}")
            # integer 0/1 only: bool, float (0.0/1.0), str and other types are rejected
            if isinstance(rel, bool) or not isinstance(rel, Integral) or rel not in (0, 1):
                raise ValueError(
                    f"relevance for ({qid!r}, {did!r}) must be 0 or 1 (an integer), got {rel!r}"
                )
    # absent key OR empty mapping both mean "no qrel rows"; a deliberate no-relevant query
    # must carry explicit zero judgments, so missing labels cannot shrink the denominator.
    missing_qrels = [q for q in query_ids if not qrels.get(q)]
    if missing_qrels:
        raise ValueError(f"queries with no qrel rows: {missing_qrels}")

    unexpected = [q for q in rankings if q not in expected]
    if unexpected:
        raise RunContractError(f"rankings contain unexpected query_id(s): {unexpected}")
    omitted = [q for q in query_ids if q not in rankings]
    if omitted:
        raise RunContractError(
            f"no ranking record for expected query(ies): {omitted} "
            "(an explicit empty list is different from an omitted query)"
        )

    per_query: list[QueryMetrics] = []
    for qid in query_ids:
        raw_ranking = rankings[qid]
        _validate_ranking(raw_ranking, corpus, f"query {qid!r}")  # validate BEFORE conversion
        ranking = list(raw_ranking)
        relevant = {d for d, r in qrels[qid].items() if r == 1}
        status = STATUS_SCORED if relevant else STATUS_NO_RELEVANT
        per_query.append(
            QueryMetrics(
                query_id=qid,
                status=status,
                n_relevant=len(relevant),
                n_returned=len(ranking),
                recall={k: recall_at_k(relevant, ranking, k) for k in ks},
                rr={k: rr_at_k(relevant, ranking, k) for k in ks},
            )
        )

    scored = [q for q in per_query if q.status == STATUS_SCORED]
    aggregate: dict[int, AggregateMetrics] = {}
    for k in ks:
        if scored:
            aggregate[k] = AggregateMetrics(
                macro_recall=math.fsum(q.recall[k] for q in scored) / len(scored),
                mrr=math.fsum(q.rr[k] for q in scored) / len(scored),
            )
        else:
            aggregate[k] = AggregateMetrics(None, None)
    return RunMetrics(
        ks=ks,
        per_query=tuple(per_query),
        aggregate=aggregate,
        total_queries=len(per_query),
        scored_queries=len(scored),
        no_relevant_queries=len(per_query) - len(scored),
    )
