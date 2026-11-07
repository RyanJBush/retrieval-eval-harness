"""Pinned real-model CPU smoke check. Exits nonzero if ANY check fails.

Run:  uv run --extra model python scripts/smoke_model.py
Needs the `model` extra and network access on first run (afterwards the pinned snapshot is cached).

The revision is pinned here as a literal and must equal retrieval_eval.retrievers.PINNED_MODEL_REVISION.
It never follows a moving branch. If the pinned revision is unavailable this FAILS; it does not fall
back to another revision or model. The printed timings are setup observations, not benchmarks.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass

EXPECTED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
EXPECTED_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
EXPECTED_DIM = 384
EXPECTED_MAX_SEQ_LENGTH = 256
EXPECTED_DEVICE = "cpu"
TEXTS = ["reset a forgotten password", "how do I recover my account login"]
# Only the files sentence-transformers actually loads (~90 MB). The repo also holds ONNX/OpenVINO/TF/Rust
# variants (~800 MB more) that this project never uses.
LOADED_FILE_PATTERNS = [
    "*.json", "*.txt", "model.safetensors", "1_Pooling/*",
]


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


def run_checks(
    model_id: str = EXPECTED_MODEL_ID,
    revision: str = EXPECTED_REVISION,
    expected_dim: int = EXPECTED_DIM,
    expected_max_seq_length: int = EXPECTED_MAX_SEQ_LENGTH,
    expected_device: str = EXPECTED_DEVICE,
) -> list[Check]:
    import numpy as np

    from retrieval_eval import retrievers

    checks: list[Check] = []

    def record(name: str, ok: bool, detail: str) -> bool:
        checks.append(Check(name, bool(ok), detail))
        return bool(ok)

    record(
        "pin_consistent_with_package",
        EXPECTED_REVISION == retrievers.PINNED_MODEL_REVISION and EXPECTED_MODEL_ID == retrievers.DEFAULT_MODEL_ID,
        f"script={EXPECTED_MODEL_ID}@{EXPECTED_REVISION[:12]} package={retrievers.DEFAULT_MODEL_ID}@{retrievers.PINNED_MODEL_REVISION[:12]}",
    )

    # 1. The exact pinned revision must be reachable (hub or local cache) and resolve to that commit.
    try:
        from huggingface_hub import snapshot_download

        t0 = time.perf_counter()
        path = snapshot_download(model_id, revision=revision, allow_patterns=LOADED_FILE_PATTERNS)
        resolved = path.rstrip("/\\").replace("\\", "/").split("/")[-1]
        record("revision_accessible", resolved == revision,
               f"snapshot dir commit={resolved} (expected {revision}); {time.perf_counter() - t0:.2f}s (observation, not a benchmark)")
    except Exception as exc:  # noqa: BLE001 - reported as a failed check, exit code is nonzero
        record("revision_accessible", False, f"{type(exc).__name__}: {exc}")
        return checks  # cannot continue without the pinned snapshot

    # 2. Load on CPU through the same wrapper the adapter uses, with the revision passed explicitly.
    try:
        t0 = time.perf_counter()
        enc = retrievers.SentenceTransformerEncoder(model_id, revision=revision, device=expected_device)
        record("model_loads", True, f"loaded in {time.perf_counter() - t0:.2f}s (observation, not a benchmark)")
    except Exception as exc:  # noqa: BLE001
        record("model_loads", False, f"{type(exc).__name__}: {exc}")
        return checks

    model = enc._model
    # sentence-transformers records the RESOLVED commit it loaded (verified manually: loading with
    # revision="main" reports the full commit hash, not the string "main"). Missing -> FAIL, not a vacuous pass.
    loaded = getattr(getattr(model, "model_card_data", None), "base_model_revision", None)
    record("loaded_commit_matches_pin", loaded == revision,
           f"model.model_card_data.base_model_revision={loaded!r} (expected {revision})")

    actual_device = str(model.device)
    param_device = next(model.parameters()).device.type
    record("device", actual_device == expected_device and param_device == expected_device,
           f"model.device={actual_device}, parameters on {param_device} (expected {expected_device})")
    record("max_seq_length", enc.max_seq_length == expected_max_seq_length,
           f"{enc.max_seq_length} (expected {expected_max_seq_length})")

    try:
        raw = enc.encode(TEXTS)
    except Exception as exc:  # noqa: BLE001
        record("encode_runs", False, f"{type(exc).__name__}: {exc}")
        return checks
    record("shape", raw.shape == (2, expected_dim), f"{raw.shape} (expected {(2, expected_dim)})")
    finite = bool(np.all(np.isfinite(raw)))
    record("finite", finite, f"all values finite: {finite}")
    norms = np.linalg.norm(raw.astype(np.float64), axis=1) if finite else np.zeros(len(TEXTS))
    record("nonzero", bool(np.all(norms > 0)), f"raw norms={np.round(norms, 6).tolist()}")
    if finite and np.all(norms > 0) and raw.ndim == 2:
        unit = np.linalg.norm(retrievers.normalize_rows(raw, what="smoke"), axis=1)
        record("unit_norm_after_normalization", bool(np.allclose(unit, 1.0, atol=1e-6)),
               f"norms after adapter normalization={unit.tolist()}")
        cos = float(retrievers.normalize_rows(raw, what="smoke")[0] @ retrievers.normalize_rows(raw, what="smoke")[1])
        record("cosine_is_finite_informational", bool(np.isfinite(cos)), f"cosine={cos:.6f} (no target value; informational)")
    else:
        record("unit_norm_after_normalization", False, "skipped: embeddings not finite/nonzero/2-D")
    return checks


def main() -> int:
    import platform

    print(f"python {platform.python_version()} on {platform.system()} {platform.machine()}")
    try:
        import sentence_transformers, torch

        print(f"torch {torch.__version__}; sentence-transformers {sentence_transformers.__version__}; torch threads {torch.get_num_threads()}")
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL environment: cannot import model dependencies ({type(exc).__name__}: {exc}); install the 'model' extra")
        return 1
    checks = run_checks()
    for c in checks:
        print(f"{'PASS' if c.ok else 'FAIL'} {c.name}: {c.detail}")
    failed = [c.name for c in checks if not c.ok]
    print(f"\n{len(checks) - len(failed)}/{len(checks)} checks passed" + (f"; FAILED: {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
