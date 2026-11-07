# Fixture checksums: which file is authoritative

| File | Status | Meaning |
|---|---|---|
| `FIXTURE.sha256` | **Authoritative** | Generated from the current `documents.jsonl`, `queries.jsonl`, `qrels.csv` on 2026-11-07 by Claude. Verify with `sha256sum -c FIXTURE.sha256` in this directory. Also checked by `tests/test_data.py::test_fixture_manifest_matches_files`. |
| `SHA256SUMS` | **Legacy evidence, non-authoritative** | Found in the project directory with a timestamp earlier than Claude's fixture writes; Claude did not create it and its origin is unknown. Kept byte-for-byte unchanged. Authorship is not inferred from timestamps. |

`SHA256SUMS` (238 bytes) has its own SHA-256 of
`798a5c9c3c87e9844e52c9102c28dc512722463a60e445f107e2018720fba47a`. Running `sha256sum -c SHA256SUMS` fails on all three lines:

| File | `SHA256SUMS` lists | Current file hashes to |
|---|---|---|
| documents.jsonl | 6a89b9b182355e9f87fb6742057c7cd14d8633441065a35f5006a1bc594b6378 | ccd017d315a31bb71c17fc1607282f555a110b544787b42be5ecf8c0f27f60c1 |
| queries.jsonl | 755d7b04b1f9651beb3cf4d1ae597c77ef6a7c836ae9f2c632d3aa14c2dfe9ed | b604a9bc9a2f62d441c30b4398d424b3ab9fa9dd011f5995c1cdf8e80691a0dd |
| qrels.csv | c5ad2c40fb13ff83c55b6f3f0b40c0251e37f35cd78cbedcfc3a1191f07e2014 | fbe98a9ba0179ccce8d862fe387b396b0e6ded272725543975a1e56c383f56d5 |

What this does not establish: whether earlier versions of the fixture files existed (the legacy hashes suggest some version
with different bytes did), or whether Claude's writes overwrote them. The current bytes are verifiable; the chronology
"labels were authored before any retriever was run" remains Claude-reported and cannot be proven by an archive.
