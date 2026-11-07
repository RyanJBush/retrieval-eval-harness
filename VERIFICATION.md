# Final delivery verification

This report records the prepublication archive checks. See [GitHub Actions](https://github.com/RyanJBush/retrieval-eval-harness/actions/workflows/tests.yml)
for hosted results. The bundled run records the source fingerprint of the [initial publication](https://github.com/RyanJBush/retrieval-eval-harness/commit/de6aa33488e70cdbdb8c06eda970ab6e2be6e3a3).
The subsequent viewer startup fix flushes the printed URL when stdout is captured. It leaves the retrieval and timing code as published.

Prepared by Claude on 2026-11-07. **All results below were produced on Linux** (Linux 6.18 x86_64, Intel Xeon 2.10 GHz, 1 logical CPU, 3.9 GiB, Python 3.12.3,
uv 0.11.32, torch 2.15.0+cpu, sentence-transformers 6.2.0, one torch thread). They are Claude-reported until an independent reviewer reruns them. **Nothing in this
package was run on Windows by Claude, and no hosted CI run exists.**

## Provenance of this slice
Base: the reviewed `retrieval-eval-review-viewer.zip` project (365 default tests). The user also supplied `retrieval-evaluation-harness-mvp.zip`, a checkpoint written by
another agent after Claude stopped. Claude reviewed it (see `evidence/uploaded_checkpoint_review.md`), adopted its four viewer corrections and its CI/`.gitattributes` files after
checking them, extended the tests, and replaced its README structure, example run and verification notes. Its Windows results are not repeated here as verified.

## The four viewer corrections (handoff 06) and the other changes
| # | Correction | Reproduced before (old code) | Verified after |
|---|---|---|---|
| 1 | Exclusive bind on Windows; bounded occupied-port regression | Not reproducible on Linux (POSIX already refuses a second listener). The Windows branch is unit-tested with a fake socket (platform `win32`: `SO_EXCLUSIVEADDRUSE` set, address reuse off; other platforms untouched) | second `make_server` on a live port raises `OSError`; `main` exits 2 with a hint; real-process test with a 30 s timeout; in-process test where `serve_forever` raises, so a bind regression fails at once instead of hanging. **Windows behavior unverified by Claude** |
| 2 | Saved ranks must equal one-based list positions | `evidence/reproduce_viewer_findings.txt`: rank 1 -> 99 accepted, "Recall@5 = 1.0" next to "missed at k=5" | rejected: `embedding/q02/d10: expected rank 1, got 99`; also int-only (bool/float/str/None rejected), duplicates, gaps, reversed lists, `n_returned`; schema 1.0 and 1.1 both open when valid and are both rejected when malformed |
| 3 | Malformed failure manifests | `{"status":"incomplete","errors":{"stage":"build"}}` -> `KeyError: 0`, exit 1, traceback | exit 2, one-line error, no server; 8 malformed shapes tested, plus a real-process test of the reported file |
| 4 | Narrow-screen overflow | real Chromium at 390x844: page scrollWidth 556 vs clientWidth 390 (horizontal page scroll), 3 tables overflowing, 0 scroll regions | scrollWidth 390 = 390, no page scroll, 4 scroll regions all scrolling locally, columns stacked, 0 console messages; desktop 1036x791 unchanged (2 columns, no overflow). `evidence/browser/` |
| + | Ctrl+C stops the viewer (README claim) | n/a | real child process: SIGINT -> exit 0, clean stderr, port closed (POSIX test). A first attempt looked like a failure only because a non-interactive shell starts background jobs with SIGINT ignored (`SigIgn: 0x6`) |
| + | CI, README, example run, `.gitattributes`, `.gitignore` | n/a | see below |

## Linux results (reported)
| Command | Result |
|---|---|
| `uv sync --locked` then `uv run --no-sync pytest -q` (base environment, no torch) | **403 passed, 12 deselected** |
| `uv sync --locked --extra model` then `uv run --no-sync pytest -m model -q` | **12 passed, 403 deselected** |
| **Unfiltered** `uv run --no-sync pytest -m 'model or not model' -q` | **415 passed** |
| `uv run --no-sync python scripts/smoke_model.py` (pinned revision `1110a243...`, CPU) | 11/11 checks, exit 0 |
| `uv run --no-sync python scripts/viewer_smoke.py examples/demo/run.json --expect-fixture-demo` | 14/14, exit 0, retrieval and model imports blocked |
| `uv run --no-sync python -m retrieval_eval.cli --dataset data/synthetic --out-dir runs/demo` (fresh) | exit 0; same run identity as the bundled example; `verify_saved_run` clean; rankings, quality and scores identical to the bundled example |

Tests per file (default + model): data 43, metrics 64, retrievers 46, retrievers_model 12, runner 63, export 33, CLI 43, provenance 7, viewer 104 = 415. All earlier tests are
unmodified; this slice added 38 default tests (viewer +31 incl. Ctrl+C, export +7). 14 mutants of this slice's code were run (Windows bind: option never set, reuse left on, helper skipped, wrong value;
occupied-port hint lost; public bind; rank not compared, bool accepted, `n_returned` not compared; `errors` not type-checked; a table not wrapped; viewport meta removed; scroll rule removed; Ctrl+C not handled): all caught.

`evidence/readme_walkthrough_log.txt` is the literal execution of every command in the README, from a clean copy of this package (no `.venv`); `evidence/readme_walkthrough_from_final_zip_log.txt` repeats it from a fresh extraction of the final ZIP (same results; its fresh run measured an embedding median of 10.791 ms against 8.441 ms in the first log, a reminder of how much a single latency figure varies). The logs cover: quick start, the demo pages (q02 k=3, q03 k=5 and k=1,
q07 k=5, q08 k=3, q01 k=1) with their key text asserted, a second viewer on the same port refused (exit 2), Ctrl+C (exit 0), default tests, viewer smoke, model tests, unfiltered suite,
model smoke, a fresh CLI run compared with the bundled example, viewer smoke on the fresh run, and the fixture manifest check afterwards.

## Bundled real run (`examples/demo/`)
Run `a2a9d02d2f2b4698a07a02c470e68c18`, schema 1.1, produced by `uv run --no-sync python -m retrieval_eval.cli --dataset data/synthetic --out-dir examples/demo` with the real pinned model on CPU; the fixture
manifest was `OK` (3/3) before and after. 8 total / 7 scored / 1 no-judged-relevant per method; 6 / 48 / 80 / 320 / 18 CSV rows; 370 repeat-search checks, 0 ranking disagreements; model cache `verified_complete`; its code
fingerprint matches the initial publication. BM25 Recall@1/3/5 0.7143, MRR 0.8571; embedding 0.8571 / 1.0000 / 1.0000, MRR 1.0000. Latency (n=160 per method): BM25 median 0.109 / p95 0.207 ms; embedding 11.833 / 16.604 ms. The run was
recorded while the machine was slow (library import 24.2 s, model load 2.9 s, document index 4.0 s, against 4 to 6 s, 0.4 to 0.5 s and 0.14 to 0.25 s in other runs); a fresh run later measured an embedding median of 8.441 ms. The quality-bearing values
are identical across runs.

## Continuous integration
`.github/workflows/tests.yml`: Ubuntu and Windows, locked base environment only, `uv run --no-sync pytest -q` plus the viewer smoke on the bundled run, 10-minute job timeout, SHA-pinned actions. Checked here: the three action SHAs
match their release tags (`git ls-remote`; the GitHub API was unavailable), the uv 0.12.17 tag exists, the YAML parses, and `actionlint` 1.7.12 reports no findings (`evidence/ci/workflow_checks.txt`); the same commands pass locally on Linux
from a clean copy. **No hosted run exists**; whether it passes on GitHub's Windows runner is unverified. The pinned CPU model smoke check is a separate manual command, not part of CI.

## Not verified / limits
- Windows: the exclusive bind, the Windows CI job, the unfiltered Windows suite, and the Windows rendering. The independent reviewer reported 376 passed (1 deselected) for the previous archive.
- The lock-file reservation on Windows or on network file systems; the cache check is a size check, not a hash check.
- Visual rendering was checked in headless Chromium only (desktop and one narrow width), not in other browsers or on a real phone.
- The synthetic fixture has seven scored queries: no claim about general retrieval quality. The label-authoring chronology is the author's statement. Actual project effort: unknown.
