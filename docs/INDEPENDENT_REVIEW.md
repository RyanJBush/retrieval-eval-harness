# Independent completion review — November 13, 2026

**Assessment: delivered for the declared synthetic-fixture retrieval-evaluation MVP.**
This review found outdated CI wording and updates documentation only. It does not change the implementation, tests, fixture, judgments, model pin, lock file or bundled results.

## Reviewed source and independence

- Repository: https://github.com/RyanJBush/retrieval-eval-harness
- Implementation reviewed: `5daa6003131c1ce3e11a1673c976c24dac2c80c7`.
- Retrieved all 54 text files through the connected GitHub service at that exact commit and verified every Git blob hash locally. Two JSON evidence files needed the original no-final-newline byte form restored after transport; final mismatches: zero.
- Binary screenshots were present in the GitHub tree but were not replayed during this review.
- This environment could not complete a fresh locked installation. The isolated Python 3.12.5 environment was populated with the exact eight locked base/dev dependency versions from the user's existing extracted wheel cache. Source was imported via `PYTHONPATH=src`.
- A local full-suite invocation was performed. It did not pass completely; the exact result and restrictions are below.
- Recomputing a saved embedding run does not independently reproduce embedding inference or prove its original execution provenance.

## Requirements and observed evidence

| Requirement | Evidence and result |
| --- | --- |
| Same inputs for both methods | Bundled run embeds one corpus, query set and qrels; independent checks confirmed 20 documents, 8 queries and all 160 query-document judgments, 9 relevant pairs |
| Lexical baseline | BM25 adapter freshly executed all 8 queries; ordered IDs and scores match the bundled run within 1e-12 |
| Embedding comparison | Pinned all-MiniLM-L6-v2 adapter and saved run inspected; fresh inference not executed because the pinned snapshot is absent |
| Metrics and denominators | Independently recalculated 96 per-query scalar values (Recall and RR over 2 methods × 8 queries × 3 cutoffs), plus aggregates; no mismatches |
| Zero versus unscored | q08 has no judged-relevant document and stays null/excluded; q02's BM25 miss remains scored as zero |
| Latency reporting | Recomputed medians and nearest-rank p95 from 320 saved nanosecond samples, 160/method; no new latency benchmark performed |
| Saved-run integrity | `verify_saved_run` returned no problems, including dataset-file hash verification |
| CSV artifacts | Verified expected row counts: summary 6; per-query 48; rankings 80; latency 320; latency summary 18 |
| Viewer rendering | q02/k3 static HTML export succeeded with the actual viewer CLI; HTTP smoke attempt was blocked by the local environment |
| Reproducibility and CI | Current published implementation has successful Ubuntu and Windows locked base CI, including viewer smoke |
| README claims | Fixture limitation is clear; obsolete “no hosted run exists” wording corrected by this documentation change |

### Recomputed fixture results

Each method has 7 scored queries out of 8; this is a developer fixture, not an independent retrieval benchmark.

| Method | Recall@1 | Recall@3 | Recall@5 | MRR@1, @3, @5 |
| --- | ---: | ---: | ---: | ---: |
| BM25 | 0.7142857143 | 0.7142857143 | 0.7142857143 | 0.8571428571 |
| Embedding, saved ranking | 0.8571428571 | 1.0000000000 | 1.0000000000 | 1.0000000000 |

The stored Linux run has BM25 median/p95 0.1094265/0.206566 ms and embedding 11.833119/16.604222 ms.
These are recomputed summaries of existing observations, not newly measured performance. No statistical significance, production relevance or real-corpus improvement is established.

## Fresh local tests — report the failures too

Python 3.12.5, Windows. Exact cache-restored versions: numpy 2.5.4, rank-bm25 0.2.2, pytest 9.2.0,
iniconfig 2.3.1, packaging 26.3, pluggy 1.7.0, Pygments 2.21.0 and colorama 0.4.6.

The isolated full default suite returned:

**390 passed, 13 failed, 1 skipped, 12 model tests deselected in 43.12 seconds; exit 1.**

All 13 failures were viewer tests attempting loopback HTTP connections. Their tracebacks ended with
`PermissionError: [WinError 10013]` at `socket.connect`. The standalone viewer smoke check met the same restriction.
No socket policy, firewall, test expectation or application implementation was changed to bypass it.

The failures cover:
- server serving every query/cutoff;
- server security headers/content type;
- nine invalid-selection cases;
- unknown paths/methods/hosts;
- the subprocess proof that the viewer works without retrieval/model imports.

The earlier ambient-Python diagnostic run lacked rank-bm25 and returned 363 passed, 40 failed, 1 skipped, 12 deselected.
It is not treated as the supported-environment result. Restoring the exact base packages eliminated the 27 dependency-related failures.

Installation attempts:
1. Plain `uv sync --locked --offline` could not write the sandbox-redirected cache.
2. A writable project-local cache and explicit Python 3.12 resolved environment creation, but offline sync could not find a cached wheel in that new cache.
3. Exact locked package contents were copied from the pre-existing normal uv wheel cache into the isolated environment. This was not a successful new `uv sync` installation.

Fresh embedding-model inference and the 12 model-marked tests remain unexecuted in this review; no substitute encoder or invented vectors were used.

## Hosted checks inspected directly

[Run 42416440729](https://github.com/RyanJBush/retrieval-eval-harness/actions/runs/42416440729)
tested the reviewed implementation commit.

| Job | Default suite | Viewer smoke |
| --- | --- | --- |
| Ubuntu | 404 passed, 12 model tests deselected | 14/14 checks passed |
| Windows | 403 passed, 1 skipped, 12 model tests deselected | 14/14 checks passed |

Both locked installation steps succeeded. The workflow runs the base environment only; it does not download or run the embedding model.
The previous initial-publication run failed; the subsequent startup-output-flush fix is present in the reviewed successful commit.
Historical passing CI is separate from the local restricted-environment result.

## Commands executed

In the isolated review checkout, with `PYTHONPATH` set to `src` and writable temporary paths:

~~~powershell
uv sync --locked --offline
# Cache permission failure.

$env:UV_CACHE_DIR = ".review-uv-cache"
uv sync --locked --offline --python "<installed Python 3.12>"
# Offline dependency availability failure; did create the isolated environment.

# After restoring the exact locked base/dev package contents from existing uv cache:
.\.venv\Scripts\python.exe -m pytest -q --basetemp=.review-pytest-locked --tb=short
# 390 passed, 13 blocked-loopback failures, 1 skipped, 12 deselected; exit 1.

py -3.12 scripts/viewer_smoke.py examples/demo/run.json --expect-fixture-demo
# Loopback connection forbidden; exit 1.

.\.venv\Scripts\python.exe -m retrieval_eval.viewer examples/demo/run.json --export-html <review-output.html> --query q02 --k 3
# Export succeeded; exit 0.
~~~

A separate review-only script used Python's `Fraction`, `statistics` and `hashlib` for the independent formula/hash checks, then
called the saved-run verifier and freshly executed the BM25 adapter. It exited 0. Summary output is committed in
[independent-review-results.json](independent-review-results.json). Its 96 metric checks are separate from the pytest count.

## Limits and licensing

- The seven scored synthetic queries do not support general retrieval-quality, business-impact or production-performance claims.
- Fixture-label chronology remains an author statement; this review checks consistency, not whether labels predated every implementation decision.
- Original model execution and earlier model-specific tests remain recorded evidence; they were not independently rerun here.
- The dependency/model license statements in the README were inspected, not independently re-audited against every upstream license.
- No root `LICENSE` file was present in the reviewed tree. A public repository is not by itself an open-source reuse license; this review does not choose a code license for the owner.
- Model-cache integrity uses file sizes, not cryptographic model-file hashes, as the README already states.
- Total actual historical effort and remaining hours under the portfolio budget are unknown.
- This was a completion/evidence review, not a comprehensive security audit or browser-rendering audit.

## Evidence-scoped resume wording

Built a Python retrieval-evaluation harness comparing BM25 and a pinned CPU embedding model on shared documents, queries and relevance judgments; exported per-query Recall@k/RR@k, aggregate Recall@k/MRR@k, raw latency samples and an inspectable local viewer.

Verified the synthetic fixture's metrics and artifact integrity, reproduced all eight BM25 rankings and scores, and confirmed locked Ubuntu/Windows CI; keep the 20-document/eight-query fixture scope with any numeric quality claim.

## Handoff record

- Objective: verify completion of the existing harness and remove stale CI claims.
- Changed files: `README.md`, `VERIFICATION.md`, `docs/INDEPENDENT_REVIEW.md`, `docs/independent-review-results.json`; documentation/evidence only.
- Commands and observed results: recorded above; full local tests did not all pass; hosted base suites and smoke checks did.
- Unresolved assumptions: fresh model execution, unrestricted local HTTP smoke, upstream license re-audit and actual prior effort remain pending/unknown.
- One specific request for Claude if this project is extended: preserve the existing fixture judgments and the null-versus-zero scoring policy; evaluate a separately identified real or held-out corpus before proposing a general retrieval-quality claim.

Stopping point: the declared MVP is delivered with explicit verification limits. No model, fixture, runtime or test code change was needed.
