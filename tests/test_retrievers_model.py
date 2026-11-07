"""REAL-MODEL tests (marker: ``model``). Excluded from the default run; they need the `model` extra,
network on first use (then the pinned snapshot is cached), and run on CPU.

    uv run --extra model pytest -m model -q

No dependency is imported at module level, so collecting this file in a basic (dev-only) environment
works; a missing dependency makes these tests FAIL when selected (they never silently skip).
These tests assert structure and validity only. They do not require any particular semantic ranking.
"""

import importlib.util
import math
import sys
from pathlib import Path

import pytest

from retrieval_eval.data import load_dataset
from retrieval_eval.retrievers import (
    PINNED_MODEL_REVISION,
    BM25Retriever,
    EmbeddingRetriever,
    SentenceTransformerEncoder,
)

pytestmark = pytest.mark.model

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "data" / "synthetic"


def _load_smoke():
    spec = importlib.util.spec_from_file_location("smoke_model", ROOT / "scripts" / "smoke_model.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["smoke_model"] = module  # @dataclass needs the module registered while it executes
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def encoder():
    return SentenceTransformerEncoder()  # defaults: pinned revision, cpu, batch 32


@pytest.fixture(scope="module")
def dataset():
    return load_dataset(FIXTURE)


@pytest.fixture(scope="module")
def retriever(dataset, encoder):
    return EmbeddingRetriever(dataset.documents, encoder)


def test_pinned_revision_constant_is_the_reported_one():
    assert PINNED_MODEL_REVISION == "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def test_encoder_requires_explicit_revision():
    with pytest.raises(ValueError, match="explicit model revision"):
        SentenceTransformerEncoder(revision="")


def test_encoder_reports_pinned_cpu_configuration(encoder):
    info = encoder.describe()
    assert info["revision"] == PINNED_MODEL_REVISION
    assert info["device_actual"] == "cpu" and info["max_seq_length"] == 256 and info["dimension"] == 384
    assert info["encode_batch_size"] == 32


def test_smoke_checks_all_pass():
    smoke = _load_smoke()
    failed = [c for c in smoke.run_checks() if not c.ok]
    assert failed == []


def test_smoke_checks_fail_when_expectations_are_wrong():
    smoke = _load_smoke()
    by_name = {c.name: c for c in smoke.run_checks(expected_dim=999)}
    assert not by_name["shape"].ok
    by_name = {c.name: c for c in smoke.run_checks(expected_max_seq_length=128)}
    assert not by_name["max_seq_length"].ok


def test_smoke_fails_on_unavailable_revision_without_falling_back():
    smoke = _load_smoke()
    checks = smoke.run_checks(revision="0" * 40)
    assert not [c for c in checks if c.name == "revision_accessible"][0].ok
    assert not any(c.name == "model_loads" for c in checks)  # stopped; no fallback revision tried


def test_smoke_main_returns_nonzero_when_any_check_fails(monkeypatch, capsys):
    smoke = _load_smoke()
    monkeypatch.setattr(smoke, "run_checks", lambda: [smoke.Check("a", True, "ok"), smoke.Check("b", False, "bad")])
    assert smoke.main() == 1
    assert "FAIL b" in capsys.readouterr().out
    monkeypatch.setattr(smoke, "run_checks", lambda: [smoke.Check("a", True, "ok")])
    assert smoke.main() == 0


def test_fixture_documents_are_within_the_models_input_limit(retriever):
    assert retriever.describe()["documents_over_max_seq_length"] == 0  # nothing is truncated


def test_real_embedding_adapter_end_to_end_on_frozen_fixture(dataset, retriever):
    n = len(dataset.documents)
    known = set(dataset.doc_ids)
    assert retriever.doc_ids == dataset.doc_ids == BM25Retriever(dataset.documents).doc_ids
    for q in dataset.queries:
        full = retriever.search(q.text, n)
        ids = [d for d, _ in full]
        assert len(ids) == len(set(ids)) == n and set(ids) == known
        assert all(type(d) is str and type(s) is float and math.isfinite(s) for d, s in full)
        assert all(-1.0 - 1e-9 <= s <= 1.0 + 1e-9 for _, s in full)  # cosine bound
        assert [s for _, s in full] == sorted((s for _, s in full), reverse=True)
        assert retriever.search(q.text, 5) == full[:5]  # top-k is a prefix of the full ordering


def test_real_embedding_adapter_is_repeatable_on_the_same_environment(dataset, retriever):
    n = len(dataset.documents)
    for q in dataset.queries:
        first = retriever.search(q.text, n)
        second = retriever.search(q.text, n)
        assert [d for d, _ in first] == [d for d, _ in second]
        assert max(abs(a - b) for (_, a), (_, b) in zip(first, second)) <= 1e-9


def test_rebuilding_the_index_reproduces_the_rankings(dataset, encoder, retriever):
    rebuilt = EmbeddingRetriever(dataset.documents, encoder)
    for q in dataset.queries:
        assert [d for d, _ in rebuilt.search(q.text, 20)] == [d for d, _ in retriever.search(q.text, 20)]


def test_cache_probe_against_the_real_hub_library_layout(tmp_path, monkeypatch):
    """Real huggingface_hub: the probe finds the real cache layout and classifies sparse stand-in snapshots correctly,
    without consulting snapshot_download. (A reproduced hub behavior: its local-only resolution returns an EMPTY folder
    as usable, and raises IncompleteSnapshotError for a snapshot missing a single file.)"""
    import huggingface_hub
    from huggingface_hub import constants

    from retrieval_eval import runner

    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.setattr(huggingface_hub, "snapshot_download",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("probe must not call snapshot_download")))
    snap = runner.pinned_snapshot_dir()
    assert snap.name == PINNED_MODEL_REVISION and snap.parent.name == "snapshots"
    assert snap.parent.parent.name == "models--sentence-transformers--all-MiniLM-L6-v2"
    assert runner.probe_pinned_model_cache()["state"] == "not_cached"
    snap.mkdir(parents=True)
    assert runner.probe_pinned_model_cache()["state"] == "partial"                   # empty directory
    for name, size in runner.PINNED_MODEL_FILES.items():
        (snap / name).parent.mkdir(parents=True, exist_ok=True)
        with open(snap / name, "wb") as fh:
            fh.truncate(size)                                                        # sparse file of the exact size
    assert runner.probe_pinned_model_cache()["state"] == "verified_complete"
    (snap / "config.json").unlink()
    one = runner.probe_pinned_model_cache()
    assert one["state"] == "partial" and one["missing"] == ["config.json"]
