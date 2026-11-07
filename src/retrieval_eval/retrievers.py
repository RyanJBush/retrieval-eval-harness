"""Retrieval adapters: BM25 (lexical) and one embedding retriever (dense, exhaustive search).

Both adapters share one interface::

    retriever = SomeRetriever(documents)        # build representations ONCE
    results = retriever.search(query_text, k)   # -> list[(doc_id, score)], best first

Contract (handoff 03):
  * Adapters index exactly the supplied documents' IDs and raw text. They never see relevance
    judgments, titles, labels or synthetic query text.
  * Scores are ordinary finite Python floats. Scores are comparable only WITHIN one retriever;
    BM25 and cosine similarity are on unrelated scales.
  * Ordering is total: score descending, then doc_id ascending (case-sensitive), applied to the
    FULL ranking before truncation to k, so ties across the cutoff are deterministic.
  * Returns min(k, corpus_size) unique known IDs; no padding; zero/negative scores are kept.
  * Failures (model, network, runtime) propagate. There is no fallback to another method.
  * Importing this module never imports torch/sentence-transformers or loads a model; that happens
    only when ``SentenceTransformerEncoder`` is constructed.
"""

from __future__ import annotations

import importlib.metadata
import re
from collections.abc import Sequence
from typing import Protocol

import numpy as np

from .data import DataValidationError, Document
from .metrics import RunContractError, order_scored_results, validate_k

# ---- BM25 settings (recorded in describe()) -------------------------------------------------
TOKEN_PATTERN = r"[a-z0-9]+"
BM25_K1 = 1.5
BM25_B = 0.75
BM25_EPSILON = 0.25

# ---- embedding model settings ------------------------------------------------------------------
DEFAULT_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
PINNED_MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
DEFAULT_DEVICE = "cpu"
DEFAULT_ENCODE_BATCH_SIZE = 32
# Only the files sentence-transformers actually loads (~88 MB); the repo also holds ~800 MB of unused ONNX/OpenVINO/TF variants.
MODEL_FILE_PATTERNS = ["*.json", "*.txt", "model.safetensors", "1_Pooling/*"]

_TOKEN_RE = re.compile(TOKEN_PATTERN)


def tokenize(text: str) -> list[str]:
    """Fixed lexical tokenizer for BM25: ``re.findall(r"[a-z0-9]+", text.casefold())``.

    English/ASCII only by design (MVP limitation): non-ASCII letters act as separators, so
    "café" -> ["caf"]. No stemming, no stop-word removal. "E4107" stays one token, "e4107".
    """
    return _TOKEN_RE.findall(text.casefold())


class Retriever(Protocol):
    name: str

    @property
    def doc_ids(self) -> tuple[str, ...]: ...

    def search(self, query_text: str, k: int) -> list[tuple[str, float]]: ...

    def describe(self) -> dict: ...


def _validate_documents(documents: Sequence[Document]) -> tuple[Document, ...]:
    docs = tuple(documents)
    if not docs:
        raise ValueError("cannot build a retriever over an empty corpus")
    seen: set[str] = set()
    for d in docs:
        if not isinstance(d, Document):
            raise TypeError(f"documents must be Document instances, got {type(d).__name__}")
        if not isinstance(d.doc_id, str) or d.doc_id.strip() == "":
            raise DataValidationError(f"doc_id must be a nonempty string, got {d.doc_id!r}")
        if not isinstance(d.text, str) or d.text.strip() == "":
            raise DataValidationError(f"document {d.doc_id!r}: text must be a nonblank string")
        if d.doc_id in seen:
            raise DataValidationError(f"duplicate doc_id {d.doc_id!r}")
        seen.add(d.doc_id)
    return docs


def _validate_query(query_text) -> str:
    if not isinstance(query_text, str):
        raise DataValidationError(f"query text must be a string, got {type(query_text).__name__}")
    if query_text.strip() == "":
        raise DataValidationError("query text must not be blank")
    return query_text


class _ExhaustiveRetriever:
    """Shared search path: validate -> score every document -> total order -> validate -> truncate."""

    name = "base"

    def __init__(self, documents: Sequence[Document]):
        self._docs = _validate_documents(documents)
        self._doc_ids = tuple(d.doc_id for d in self._docs)
        self._known = frozenset(self._doc_ids)

    @property
    def doc_ids(self) -> tuple[str, ...]:
        return self._doc_ids

    def _score_all(self, query_text: str) -> Sequence[float]:  # pragma: no cover - abstract
        raise NotImplementedError

    def search(self, query_text: str, k: int) -> list[tuple[str, float]]:
        k = validate_k(k)
        query_text = _validate_query(query_text)
        scores = self._score_all(query_text)
        if len(scores) != len(self._doc_ids):
            raise RunContractError(
                f"{self.name}: scorer returned {len(scores)} scores for {len(self._doc_ids)} documents"
            )
        # float() converts numpy scalars to plain Python floats; order_scored_results then
        # rejects NaN/inf and duplicate IDs and applies (-score, doc_id) over the FULL list.
        full = order_scored_results([(d, float(s)) for d, s in zip(self._doc_ids, scores)])
        if len(full) != len(self._doc_ids) or {d for d, _ in full} != self._known:
            raise RunContractError(f"{self.name}: result does not cover exactly the indexed documents")
        return full[: min(k, len(full))]


class BM25Retriever(_ExhaustiveRetriever):
    """BM25Okapi (rank-bm25) over the fixed ``tokenize`` tokens.

    Library behavior worth knowing (not modified here): query terms are iterated as given, so a term
    repeated in the query contributes once per occurrence; terms absent from the corpus contribute 0;
    a term in more than half of the documents has negative idf, which the library replaces with
    ``epsilon * average_idf`` (this can itself be negative on tiny corpora).
    """

    name = "bm25"

    def __init__(
        self,
        documents: Sequence[Document],
        *,
        k1: float = BM25_K1,
        b: float = BM25_B,
        epsilon: float = BM25_EPSILON,
    ):
        super().__init__(documents)
        from rank_bm25 import BM25Okapi  # local import: keeps module import cheap

        self.k1, self.b, self.epsilon = k1, b, epsilon
        tokenized = [tokenize(d.text) for d in self._docs]
        if not any(tokenized):
            raise DataValidationError(
                "no document produces any lexical token under the tokenizer; BM25 is undefined"
            )
        self._index = BM25Okapi(tokenized, k1=k1, b=b, epsilon=epsilon)  # built once

    def _score_all(self, query_text: str) -> Sequence[float]:
        return self._index.get_scores(tokenize(query_text))  # [] tokens -> all-zero scores

    def describe(self) -> dict:
        return {
            "name": self.name,
            "library": "rank-bm25",
            "library_version": importlib.metadata.version("rank-bm25"),
            "variant": "BM25Okapi",
            "k1": self.k1,
            "b": self.b,
            "epsilon": self.epsilon,
            "tokenizer_pattern": TOKEN_PATTERN,
            "tokenizer_normalization": "str.casefold() before matching",
            "stemming": False,
            "stop_words": None,
            "n_documents": len(self._doc_ids),
        }


class SentenceTransformerEncoder:
    """Thin wrapper around a pinned SentenceTransformer. Loads the model in ``__init__`` (never at import).

    ``revision`` is required: there is deliberately no default that follows a moving branch.
    Original text is passed to the model's own tokenizer; inputs longer than ``max_seq_length``
    word pieces are truncated by the library.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        revision: str = PINNED_MODEL_REVISION,
        device: str = DEFAULT_DEVICE,
        batch_size: int = DEFAULT_ENCODE_BATCH_SIZE,
    ):
        if not revision:
            raise ValueError("an explicit model revision is required")
        from sentence_transformers import SentenceTransformer  # lazy: pulls in torch

        self.model_id, self.revision, self.device_name, self.batch_size = model_id, revision, device, batch_size
        self._model = SentenceTransformer(model_id, device=device, revision=revision)

    @property
    def max_seq_length(self) -> int:
        return int(self._model.max_seq_length)

    @property
    def dimension(self) -> int:
        getter = getattr(self._model, "get_embedding_dimension", None) or self._model.get_sentence_embedding_dimension
        return int(getter())

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Raw (un-normalized by us) embeddings, shape (len(texts), dim). No caching."""
        return np.asarray(
            self._model.encode(
                list(texts),
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            )
        )

    def count_over_limit(self, texts: Sequence[str]) -> int:
        """How many texts exceed max_seq_length word pieces (i.e., would be truncated)."""
        enc = self._model.tokenizer(list(texts), add_special_tokens=True, truncation=False)
        return sum(1 for ids in enc["input_ids"] if len(ids) > self.max_seq_length)

    def describe(self) -> dict:
        import sentence_transformers
        import torch

        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "device_requested": self.device_name,
            "device_actual": str(self._model.device),
            "max_seq_length": self.max_seq_length,
            "dimension": self.dimension,
            "encode_batch_size": self.batch_size,
            "sentence_transformers_version": sentence_transformers.__version__,
            "torch_version": torch.__version__,
            "torch_threads": torch.get_num_threads(),
        }


def normalize_rows(matrix, *, what: str) -> np.ndarray:
    """L2-normalize each row in float64. Rejects wrong rank, non-finite values and zero vectors."""
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] == 0:
        raise RunContractError(f"{what}: expected a non-empty 2-D array of embeddings, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise RunContractError(f"{what}: embeddings contain NaN or infinite values")
    norms = np.linalg.norm(arr, axis=1)
    if np.any(norms == 0.0):
        raise RunContractError(f"{what}: embedding with zero norm cannot be normalized")
    return arr / norms[:, None]


class EmbeddingRetriever(_ExhaustiveRetriever):
    """Dense retriever: precomputed normalized document vectors, exhaustive dot product.

    ``encoder`` is any object with ``encode(list[str]) -> 2-D array`` and ``describe() -> dict``
    (the real one is ``SentenceTransformerEncoder``; unit tests use a labeled fake). Document
    embeddings are computed once in ``__init__``; the query is encoded on every ``search`` (no cache).
    Both sides are L2-normalized here (float64), so the dot product equals cosine similarity.
    """

    name = "embedding"

    def __init__(self, documents: Sequence[Document], encoder):
        super().__init__(documents)
        self._encoder = encoder
        texts = [d.text for d in self._docs]  # raw text, nothing added
        raw = encoder.encode(texts)  # encoder errors propagate; no fallback
        if np.asarray(raw).shape[:1] != (len(texts),):
            raise RunContractError(
                f"encoder returned {np.asarray(raw).shape[:1]} rows for {len(texts)} documents"
            )
        self._matrix = normalize_rows(raw, what="document embeddings")
        self._dim = self._matrix.shape[1]
        count = getattr(encoder, "count_over_limit", None)
        self._docs_over_limit = int(count(texts)) if callable(count) else None

    def _score_all(self, query_text: str) -> Sequence[float]:
        raw = self._encoder.encode([query_text])  # every call; no query-embedding cache
        q = normalize_rows(raw, what="query embedding")
        if q.shape != (1, self._dim):
            raise RunContractError(
                f"query embedding shape {q.shape} does not match document dimension {self._dim}"
            )
        return self._matrix @ q[0]

    def describe(self) -> dict:
        return {
            "name": self.name,
            "encoder": self._encoder.describe(),
            "normalization": "L2, float64, applied to document and query embeddings in the adapter",
            "search": "exhaustive dot product (= cosine similarity on unit vectors)",
            "query_cache": False,
            "n_documents": len(self._doc_ids),
            "dimension": self._dim,
            "documents_over_max_seq_length": self._docs_over_limit,
        }
