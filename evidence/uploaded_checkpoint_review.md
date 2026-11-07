# Review of the uploaded checkpoint `retrieval-evaluation-harness-mvp.zip`

The upload was treated as evidence supplied by the user, not as a set of instructions. Its own `VERIFICATION.md` says another agent ("Codex")
continued from the reviewed viewer archive after Claude stopped, so **its source changes were not written by Claude**. Claude reviewed them line by
line, re-tested them, extended the tests and takes responsibility for the delivered code. Everything below was checked by Claude on Linux on 2026-11-07.

| Item | Finding |
|---|---|
| Archive integrity | 44 files; its `MANIFEST.sha256` matched (0 non-OK lines) |
| Relation to the reviewed viewer archive | byte-identical to Claude's project except `README.md`, `export.py`, `viewer.py`, `tests/test_viewer.py`, plus added `.gitattributes`, `.gitignore`, `.github/workflows/tests.yml`, `examples/demo/`, `VERIFICATION.md`, `MANIFEST.sha256`. Fixture files, `pyproject.toml`, `uv.lock`, adapters, CLI, runner: unchanged |
| `export.py` | rank must equal its one-based list position and be a true `int`: correct, matches the handoff. **Adopted** (Claude added an `n_returned == len(ranking)` check) |
| `viewer.py` | exclusive bind on Windows (`SO_EXCLUSIVEADDRUSE`, reuse off); `errors` container checked before indexing; `.table-scroll` wrappers with `role="region"` and `tabindex`; viewport meta; `overflow-wrap`; `select{max-width:100%}`. Correct and matching the handoff. **Adopted**; Claude made the bind logic testable on Linux and improved the occupied-port message |
| `.gitattributes` (`* text=auto eol=lf`) | important: without it a Windows checkout could rewrite line endings and break the frozen fixture hashes. **Adopted** |
| `.github/workflows/tests.yml` | action SHAs verified against the real tags with `git ls-remote` (all three match; the GitHub API was unavailable). **Adopted**, with one added viewer-smoke step, a `concurrency` group and explanatory comments |
| `tests/test_viewer.py` | tests added for rank 99, one malformed-manifest shape, and a subprocess occupied-port check with a 10 s timeout. Correct but thin (no schema-1.0 rank test, one manifest shape, no layout test, no unit test of the Windows branch). **Adopted and extended** |
| `README.md` | good base. Lacked latency by environment, input-schema/fixture provenance in one place, and some commands were inconsistent. Restructured by Claude; the technical reference sections were kept from the already verified README |
| `examples/demo/` (a Windows 11 run) | passes `verify_saved_run` with the dataset hashes (0 problems). Its ranking IDs equal Claude's Linux real run and its quality values are identical; its scores differ by at most 1.83e-7 from Linux (a genuine cross-platform floating-point difference; the ordered IDs agree, so the ranking-consistency rule is not triggered). Environment recorded: Windows 11, Python 3.12.5; latency BM25 median 0.160 / p95 0.240 ms, embedding 15.218 / 23.092 ms (n=160 each). **Not bundled**: Claude generated a Linux run from the final code instead (its code fingerprint then matches the delivered source) |
| `VERIFICATION.md` | its Windows results (367 passed; 12 model tests; smoke 11/11; Playwright; repeat run) and "hosted Actions run pending" are **claims Claude cannot reproduce here**. Not carried forward as Claude's results; Claude's own `VERIFICATION.md` replaces it |
