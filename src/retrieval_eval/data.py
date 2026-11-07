"""Input schema loading and validation for the retrieval evaluation harness.

Layout of a dataset directory:
    documents.jsonl  one JSON object per line: {"doc_id": str, "text": str}
    queries.jsonl    one JSON object per line: {"query_id": str, "text": str}
    qrels.csv        header ``query_id,doc_id,relevance``; relevance is exactly 0 or 1

Policy (see handoff section 4): IDs are case-sensitive, preserved exactly, never
stripped or normalised. Repeated keys inside a JSON object are rejected, not overwritten. Any violation raises ``DataValidationError`` with context.
Fully blank lines in JSONL/CSV files are ignored; everything else is strict.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

QRELS_HEADER = ["query_id", "doc_id", "relevance"]
_MAX_LISTED = 5


class DataValidationError(ValueError):
    """Raised for any malformed or inconsistent input."""


class _DuplicateKeyError(ValueError):
    """Internal: a JSON object repeated a key. Converted to DataValidationError with context."""


def _reject_duplicate_keys(pairs):
    """object_pairs_hook: json.loads would otherwise keep only the last repeated key."""
    seen: set[str] = set()
    for key, _ in pairs:
        if key in seen:
            raise _DuplicateKeyError(key)
        seen.add(key)
    return dict(pairs)


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str


@dataclass(frozen=True)
class Query:
    query_id: str
    text: str


@dataclass(frozen=True)
class Dataset:
    documents: tuple[Document, ...]
    queries: tuple[Query, ...]
    qrels: dict[str, dict[str, int]]  # query_id -> {doc_id: 0 | 1}

    @property
    def doc_ids(self) -> tuple[str, ...]:
        return tuple(d.doc_id for d in self.documents)

    @property
    def query_ids(self) -> tuple[str, ...]:
        return tuple(q.query_id for q in self.queries)

    def relevant_ids(self, query_id: str) -> frozenset[str]:
        return frozenset(d for d, r in self.qrels[query_id].items() if r == 1)

    def label(self, query_id: str, doc_id: str) -> str:
        """'relevant', 'nonrelevant' (explicit 0), or 'unjudged' (no qrel row)."""
        rel = self.qrels[query_id].get(doc_id)
        if rel is None:
            return "unjudged"
        return "relevant" if rel == 1 else "nonrelevant"

    def missing_judgments(self) -> list[tuple[str, str]]:
        """(query_id, doc_id) pairs with no qrel row. Empty for a complete fixture."""
        return [
            (q, d)
            for q in self.query_ids
            for d in self.doc_ids
            if d not in self.qrels[q]
        ]


def _fmt(items) -> str:
    items = list(items)
    shown = ", ".join(repr(i) for i in items[:_MAX_LISTED])
    more = f" (+{len(items) - _MAX_LISTED} more)" if len(items) > _MAX_LISTED else ""
    return shown + more


def _check_id(value, where: str, field: str) -> str:
    if not isinstance(value, str):
        raise DataValidationError(
            f"{where}: {field} must be a string, got {type(value).__name__} ({value!r})"
        )
    if value.strip() == "":
        raise DataValidationError(f"{where}: {field} must be a nonempty string")
    return value


def _check_text(value, where: str) -> str:
    if not isinstance(value, str):
        raise DataValidationError(
            f"{where}: text must be a string, got {type(value).__name__}"
        )
    if value.strip() == "":
        raise DataValidationError(f"{where}: text must not be blank")
    return value


def _read_jsonl(path: Path, id_field: str) -> list[tuple[str, str]]:
    path = Path(path)
    if not path.is_file():
        raise DataValidationError(f"{path}: file not found")
    expected = {id_field, "text"}
    rows: list[tuple[str, str]] = []
    seen: set[str] = set()
    try:
        with open(path, encoding="utf-8", newline="") as fh:
            for lineno, line in enumerate(fh, 1):
                if line.strip() == "":
                    continue
                where = f"{path}:{lineno}"
                try:
                    obj = json.loads(line, object_pairs_hook=_reject_duplicate_keys)
                except _DuplicateKeyError as exc:
                    raise DataValidationError(
                        f"{where}: duplicate key {exc.args[0]!r} in JSON object"
                    ) from exc
                except json.JSONDecodeError as exc:
                    raise DataValidationError(f"{where}: invalid JSON ({exc.msg})") from exc
                if not isinstance(obj, dict):
                    raise DataValidationError(f"{where}: expected a JSON object")
                missing = expected - obj.keys()
                extra = obj.keys() - expected
                if missing:
                    raise DataValidationError(f"{where}: missing field(s) {sorted(missing)}")
                if extra:
                    raise DataValidationError(f"{where}: unexpected field(s) {sorted(extra)}")
                ident = _check_id(obj[id_field], where, id_field)
                text = _check_text(obj["text"], where)
                if ident in seen:
                    raise DataValidationError(f"{where}: duplicate {id_field} {ident!r}")
                seen.add(ident)
                rows.append((ident, text))
    except UnicodeDecodeError as exc:
        raise DataValidationError(f"{path}: not valid UTF-8 ({exc.reason})") from exc
    return rows


def load_documents(path) -> tuple[Document, ...]:
    rows = _read_jsonl(Path(path), "doc_id")
    if not rows:
        raise DataValidationError(f"{path}: corpus is empty")
    return tuple(Document(i, t) for i, t in rows)


def load_queries(path) -> tuple[Query, ...]:
    rows = _read_jsonl(Path(path), "query_id")
    if not rows:
        raise DataValidationError(f"{path}: query set is empty")
    return tuple(Query(i, t) for i, t in rows)


def load_qrels(path) -> dict[str, dict[str, int]]:
    path = Path(path)
    if not path.is_file():
        raise DataValidationError(f"{path}: file not found")
    qrels: dict[str, dict[str, int]] = {}
    try:
        with open(path, encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh)
            header = next(reader, None)
            if header != QRELS_HEADER:
                raise DataValidationError(
                    f"{path}:1: header must be exactly {','.join(QRELS_HEADER)}, got {header!r}"
                )
            for row in reader:
                where = f"{path}:{reader.line_num}"
                if not row:
                    continue
                if len(row) != 3:
                    raise DataValidationError(
                        f"{where}: expected 3 columns, got {len(row)}"
                    )
                qid = _check_id(row[0], where, "query_id")
                did = _check_id(row[1], where, "doc_id")
                if row[2] not in ("0", "1"):
                    raise DataValidationError(
                        f"{where}: relevance must be exactly 0 or 1, got {row[2]!r}"
                    )
                per_q = qrels.setdefault(qid, {})
                if did in per_q:
                    raise DataValidationError(
                        f"{where}: duplicate qrel pair ({qid!r}, {did!r})"
                    )
                per_q[did] = int(row[2])
    except UnicodeDecodeError as exc:
        raise DataValidationError(f"{path}: not valid UTF-8 ({exc.reason})") from exc
    return qrels


def validate_references(
    documents: tuple[Document, ...],
    queries: tuple[Query, ...],
    qrels: dict[str, dict[str, int]],
) -> None:
    doc_ids = {d.doc_id for d in documents}
    query_ids = {q.query_id for q in queries}
    unknown_q = [q for q in qrels if q not in query_ids]
    if unknown_q:
        raise DataValidationError(f"qrels reference unknown query_id(s): {_fmt(unknown_q)}")
    unknown_d = sorted({d for per_q in qrels.values() for d in per_q if d not in doc_ids})
    if unknown_d:
        raise DataValidationError(f"qrels reference unknown doc_id(s): {_fmt(unknown_d)}")
    unlabeled = [q.query_id for q in queries if q.query_id not in qrels]
    if unlabeled:
        raise DataValidationError(
            f"query(ies) with no qrel rows: {_fmt(unlabeled)} "
            "(a deliberate no-relevant query needs explicit 0 rows)"
        )


def load_dataset(directory) -> Dataset:
    directory = Path(directory)
    documents = load_documents(directory / "documents.jsonl")
    queries = load_queries(directory / "queries.jsonl")
    qrels = load_qrels(directory / "qrels.csv")
    validate_references(documents, queries, qrels)
    return Dataset(documents=documents, queries=queries, qrels=qrels)
