# Retrieval Evaluation Harness

[![Tests](https://github.com/RyanJBush/retrieval-eval-harness/actions/workflows/tests.yml/badge.svg)](https://github.com/RyanJBush/retrieval-eval-harness/actions/workflows/tests.yml)

Compares BM25 with a pinned CPU embedding model (`sentence-transformers/all-MiniLM-L6-v2`) on the same documents, queries and
relevance judgments. The CLI exports per-query Recall@k and RR@k, aggregate Recall@k and MRR@k, and CPU retrieval latency
in JSON and CSV. The local viewer shows both rankings side by side so you can inspect what each method missed.

The bundled data is a small **synthetic developer fixture** (20 documents, 8 queries). The results demonstrate that the pipeline
and the viewer work; they are not a benchmark and support no claim about retrieval quality on real data.

## Open the bundled example

Needs Python 3.12 and [uv](https://docs.astral.sh/uv/). Installing dependencies can require internet access.
Once setup is complete, the viewer opens the bundled run offline and uses the texts and judgments embedded in that file.

```console
uv sync --locked
uv run --no-sync python -m retrieval_eval.viewer examples/demo/run.json --port 0
```

Open the printed `http://127.0.0.1:PORT/` address. **Stop the viewer with Ctrl+C** in the terminal. `--port 0` picks any free
port; leave it out for the default 8765, or pass `--port N`. If the port is taken the viewer exits with code 2 and says so.
`examples/demo/` is a complete real run of the CLI (`run.json` plus five CSVs) on Linux with the real model; the viewer reads only
`run.json`, which embeds the document and query texts and every judgment.

[See the reimbursement query side by side](evidence/browser/after_desktop_q02_k3.png) or read the [verification notes](VERIFICATION.md).

### Two-minute demo

In the page, pick a query and a cutoff k, then press **Show**.

1. **Top of the page.** Both methods have 7 scored queries out of 8. At k=5, BM25 has macro Recall 0.7143 and MRR 0.8571;
   the embedding model has 1.0000 on both. The latency table lists median and p95 in milliseconds with 160 measured searches per method.
2. **q02, k=3 (a wording miss).** The relevant document `d10` says travel costs are *repaid*; the query says *reimbursed*, and after
   BM25 tokenization they share no word. BM25's saved top five miss it, so Recall@3 is 0 (shown as a miss, not as unscored). The embedding
   model ranks `d10` first.
3. **q03 or q07, k=5 (second relevant document missed).** Both methods put the first relevant document at rank 1. BM25 misses the second
   one (`d02`, which shares only the word "to" with the q03 query; `d19`, which shares nothing with the q07 query). The embedding model
   finds it at rank 2, so its Recall is 1.0 against BM25's 0.5 while reciprocal rank is 1.0 for both. At k=1 the embedding view
   shows the second document as "missed at k=1" with its rank in the saved top five.
4. **q08, k=3 (unscored).** The query has no judged-relevant document, so Recall and RR are null: the page says "unscored", not zero.
   It is excluded from the macro means and counted separately.
5. **q01, k=1 (exact identifier).** The query `E4107` is an exact error code; both methods rank `d01` first.

## What the frozen synthetic results show

One saved run of the fixture (`examples/demo`, 8 queries, 7 with at least one judged-relevant document):

| Method | Recall@1 | Recall@3 | Recall@5 | MRR@1, @3, @5 | Queries scored / total |
|---|---:|---:|---:|---:|---:|
| BM25 | 0.7143 | 0.7143 | 0.7143 | 0.8571 | 7 / 8 |
| Embedding | 0.8571 | 1.0000 | 1.0000 | 1.0000 | 7 / 8 |

Macro means cover the seven scored queries. The eighth (q08) stays visible with null metrics. A single scored query can change
a macro mean by up to 1/7. Treat these figures as observations about this developer fixture.

**Retrieval latency by measured environment** (warm, serial, CPU, one thread; 3 discarded warm-up searches, then 20 measured searches
per query and method, so n = 160 per method; time from raw query text to ranked results):

| Environment | BM25 median / p95 | Embedding median / p95 | Source |
|---|---:|---:|---|
| Linux x86_64, Intel Xeon 2.10 GHz, 1 logical CPU, 3.9 GiB, Python 3.12.3, torch 2.15.0+cpu | 0.109 / 0.207 ms | 11.833 / 16.604 ms | bundled run `examples/demo` |
| Windows 11, Python 3.12.5, torch 2.15.0+cpu (independent review, two runs) | 0.073, 0.074 / 0.099, 0.099 ms | 5.676, 5.890 / 6.560, 6.608 ms | reported by the independent reviewer; not bundled |

Timings vary between runs and machines. In the Linux environment above, the bundled run's embedding median is 11.8 ms and a fresh run of
the same command later the same day measured 8.4 ms (`evidence/readme_walkthrough_log.txt`). The bundled run was recorded while that
machine was visibly slow (its saved setup observations show a 24 s library import against 4 to 6 s in other runs). Repeated searches of one
query are not independent samples, so no significance is implied, and no latency target is imposed.

## Setup

Python 3.12 (pinned by `.python-version` and `requires-python`). CPU only.

```console
uv sync --locked                    # base environment: default tests, viewer, example (no torch)
uv sync --locked --extra model      # adds torch and sentence-transformers for real evaluations
```

`uv.lock` pins every dependency. Linux resolves torch from the PyTorch CPU index; Windows uses the default PyPI wheel and the adapter
selects CPU explicitly. The pinned model snapshot (revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, Apache-2.0 per its model card)
downloads on first use (about 90 MB) and is then cached. All commands below use `uv run --no-sync`, which uses the environment you
just synced (a later plain `uv sync --locked` removes the model packages again).

## Commands

| Purpose | Command | Needs model/network |
|---|---|---|
| Default tests (data, metrics, adapters with fakes, runner, exports, CLI, viewer) | `uv run --no-sync pytest -q` | no |
| Viewer smoke check on a copied run, retrieval and model imports blocked | `uv run --no-sync python scripts/viewer_smoke.py examples/demo/run.json --expect-fixture-demo` | no |
| Real-model tests (marker `model`) | `uv run --no-sync pytest -m model -q` | yes (first run), needs the `model` extra |
| **Pinned CPU model smoke check** (exits nonzero on any failed check) | `uv run --no-sync python scripts/smoke_model.py` | yes (first run), needs the `model` extra |

The default run deselects `model` tests (`addopts = -m 'not model'`); `-m model` on the command line overrides it. Importing
`retrieval_eval.retrievers` never loads torch or a model.

## Run an evaluation

```console
uv sync --locked --extra model
uv run --no-sync python -m retrieval_eval.cli --dataset data/synthetic --out-dir runs/demo
uv run --no-sync python -m retrieval_eval.viewer runs/demo/run.json --port 0
```

Defaults: `--k 1,3,5`, `--warmup 3`, `--repetitions 20`, `--threads 1`, `--dataset-designation synthetic`. The output directory must
not exist or must be empty; existing outputs are never overwritten. With the model already cached, a full run took roughly 7 to 35 seconds in the Linux environment above.

| Exit code | Meaning |
|---|---|
| 0 | run completed; all artifacts written |
| 1 | run or export failed, or the ordered document IDs of a repeated search changed: incomplete. `run_failed.json` (stage, method, query, error, diagnostic) is written; `run.json` and the CSVs are **not** |
| 2 | invalid configuration, dataset or output path, or an output directory that is not empty or is reserved by another invocation; rejected before any model loading; nothing written (and another invocation's files are left untouched) |

A run is complete if and only if `run.json` exists: it is written last, after every CSV. Nothing is skipped, substituted
or turned into an empty ranking; the first failure aborts the run.

### Input data and fixture provenance

A dataset directory holds three UTF-8 files. IDs are case-sensitive and kept exactly; unknown fields, duplicate keys and duplicate IDs are rejected.

| File | Content |
|---|---|
| `documents.jsonl` | one JSON object per line: `{"doc_id": str, "text": str}`; IDs unique, text non-blank |
| `queries.jsonl` | one JSON object per line: `{"query_id": str, "text": str}` |
| `qrels.csv` | header `query_id,doc_id,relevance`; relevance is exactly `0` or `1`; every query needs at least one row; a document with no row for a query is *unjudged* |

`data/synthetic/` was written by hand for this project: 20 short English documents, 8 queries and 160 explicit judgments (9 relevant, complete
for every query-document pair). It covers an exact identifier, paraphrases, ambiguous words (python, jaguar, bank), two queries with several
relevant documents, and one query with no relevant document. Per-query rationales are in `data/synthetic/README.md`. The judgments were
written before any retriever was run on the fixture (a statement by the author; an archive cannot prove it) and an independent review kept them.
`data/synthetic/FIXTURE.sha256` is the authoritative checksum list; the CLI refuses to run if the fixture no longer matches it. The legacy
`SHA256SUMS` is an unexplained file kept as evidence: it does not match the fixture (see `data/synthetic/CHECKSUMS.md`). The repository's
`.gitattributes` forces LF line endings so the checksums also hold in a Windows checkout.

### Metrics policy

For a query with judged-relevant set R and ranked list L: **Recall@k** = |R ∩ top-k of L| / |R| (the denominator is always |R|, even when
fewer than k results exist); **RR@k** = 1/r for the first relevant document at rank r ≤ k, else 0. **Macro Recall@k and MRR@k** are
arithmetic means over queries with at least one judged-relevant document. A query with none is *unscored*: its metrics are null (blank in CSV,
never zero), it is excluded from the macro means and reported in separate counts, and both methods use identical denominators. An answerable query
that returns an empty ranking scores 0 and stays in the denominator. Ties are broken by score descending, then `doc_id` ascending.

### Timing boundaries

Each measured search is timed with `perf_counter_ns` around exactly `retriever.search(raw_query_text, k=max_k)`: query tokenization or encoding,
scoring, ordering and top-k. Data reading, metric computation, result validation, disk writes and document-text lookup are outside the timed
region. Setup (library import, snapshot fetch, model load, BM25 build, document embedding) and the first call after each build are recorded
separately and are not retrieval latency. Details are in "Timing protocol" below.


### Output directory ownership

Each invocation takes **exclusive ownership of its output directory before the first build** and keeps it until every artifact
(or the failure manifest) is written. The directory is created if needed; an existing empty directory is accepted. Ownership
is the atomic creation (`O_CREAT | O_EXCL`) of `.retrieval_eval.lock` inside it, so of any number of concurrent invocations
exactly one proceeds and the others exit 2 without writing anything. After acquiring the lock the directory is checked again
(contents other than the lock must be absent), because a previous owner may have finished between a competitor's early check
and its reservation attempt; that competitor releases its own lock and leaves the finished run untouched. Only the owner
removes the lock, in a `finally` (success, failure, export failure or interruption). A lock left behind by a killed process is
not removed automatically, because it cannot be told apart from a live run: the error message says to delete it and retry.
`run.json` remains the completion marker. This relies on `O_EXCL` being atomic on a local file system (not guaranteed on some
network file systems).

### Ranking consistency rule

Every first-call, warm-up and measured search is compared with the untimed quality-pass ranking of the same (method, query).
If the **ordered document IDs differ**, the run is incomplete (stage `consistency`): `run_failed.json` keeps the diagnostic (phase,
method, query, call index, expected and observed IDs; at most 50 details, exact total) and no `run.json`, CSV or aggregate is
published. If the IDs match but scores differ, the run may complete and the differences are reported separately in
`run.json` under `ranking_consistency` and per query in `measured_search_consistency` (`n_score_differs_same_ranking`,
`max_abs_score_difference`).

### Artifacts

`run.json` is the authoritative record (schema version `1.0`): manifest and configuration, dataset file SHA-256s, the
evaluated document and query texts and all judgments, policies, adapter settings, model identity/revision/license
reference, Python/OS/CPU/RAM/thread details, every installed dependency version, code identity (git commit and dirty state,
or a source fingerprint when git has no commit), setup and first-call observations, ranked results with labels, per-query
metrics, aggregates and counts, and every raw latency sample. The CSVs are projections of it:

| File | One row per | Rows (default) | Columns |
|---|---|---|---|
| `summary.csv` | method, k | 6 | `method,k,macro_recall,mrr,total_queries,scored_queries,no_judged_relevant_queries` |
| `per_query.csv` | method, query, k | 48 | `method,query_id,query_text,k,status,n_relevant,n_returned,recall,rr` |
| `rankings.csv` | method, query, rank (max k) | 80 | `method,query_id,rank,doc_id,score,label` |
| `latency.csv` | measured search | 320 | `method,query_id,repetition,execution_index,position_in_pair,elapsed_ns` |
| `latency_summary.csv` | method overall, or method and query | 18 | `method,scope,query_id,n_samples,median_ns,p95_ns` |

Conventions: UTF-8, `\n` line endings, standard CSV quoting. **A blank cell is null and is never a zero**: a query with no
judged-relevant document has blank `recall`/`rr` and status `no_judged_relevant`; an answerable query with an empty ranking
has `0.0`. Floats are written with Python `repr` (exact round trip). `*_ns` columns are integer nanoseconds from
`time.perf_counter_ns`. `label` is `relevant`, `nonrelevant` (explicit 0 judgment) or `unjudged`. JSON uses `null` and
rejects NaN/Infinity. Rankings are saved at the maximum k; smaller cutoffs are derived from them.

`export.verify_saved_run(run, dataset_dir=None)` recomputes every metric, aggregate, label and latency summary from the saved
raw records and (optionally) re-hashes the dataset files; `export.compare_runs(a, b)` compares two runs.

### Setup observation fields (`run.json` -> `setup_observations.embedding`, schema 1.1)

The model cache state is **verified, not inferred from the hub library**. `snapshot_download(local_files_only=True)` is not a
reliable probe: with the locked `huggingface_hub` it returns an *empty* or *truncated* snapshot folder as if it were usable, and
raises `IncompleteSnapshotError` for a snapshot that merely lacks one file. The runner therefore inspects the cache directory
for the pinned revision itself and checks the presence and exact byte size of each of the 11 files selected by the download
patterns against a table recorded for the pinned revision (a size check, not a content hash).

| Field | Meaning |
|---|---|
| `model_cache_state_before_run` | `verified_complete` (all 11 files present with the expected size), `partial` (snapshot directory exists but files are missing or have the wrong size, including an empty folder), `not_cached` (no snapshot directory for the pinned revision), or `unknown` (the state could not be established) |
| `model_cache_verification` | the evidence: files expected / present with expected size, `missing`, `size_mismatch`, the verification method |
| `model_snapshot_fetch_ns` | duration of the snapshot fetch step, in integer nanoseconds. Measured separately from model load |
| `model_snapshot_fetch_may_include_download` | `false` only for `verified_complete`; `true` for `partial`, `not_cached` and `unknown`. When `true`, the fetch time may include downloading and is not a cached-setup figure |
| `model_snapshot_fetch_note` | a sentence stating exactly what the previous field means for this run |
| `library_import_ns`, `model_load_ns`, `document_embedding_index_build_ns` | first-use library imports, model load, and document embedding, each separate |

**Change from schema 1.0:** `model_cache_state_before_run` was `cached` or `not_cached`, derived from the unreliable probe, and
`model_snapshot_fetch_includes_download` was a boolean that could be wrong for an empty or partial snapshot. The latter is
replaced by `model_snapshot_fetch_may_include_download`, and the cache state now has four values with a verification record.
`schema_version` is `1.1`. Runs with schema `1.0` stay readable (`export.verify_saved_run` and the viewer accept both; the
viewer does not display setup observations).

### Timing protocol

CPU only, fixed thread count (`--threads`, default 1; OMP/MKL/OpenBLAS env vars and torch threads), methods run serially,
no query or result caches. For each of the queries, in dataset order: 3 discarded warm-up searches per method (listed method
order, results recorded separately), then 20 measured searches per method. In measured repetition *i* the method order is
rotated by *i* mod 2, so the first method alternates. Each measured search is timed with `perf_counter_ns` around exactly
`retriever.search(raw_query_text, k=max_k)`: query tokenization/encoding, scoring, ordering and top-k. Data reading, metric
computation, result validation, disk writes and text lookup are outside the timed region. Setup (library import, model
snapshot fetch, model load, BM25 build, document embedding) is recorded separately; the fetch is labeled download-inclusive
only if the model was not already cached. The first retrieval call after each index build (query `q01`) is recorded
separately and is not a cold-machine measurement. Median of an even count = mean of the two middle values; p95 = nearest rank,
the sorted value at rank ceil(0.95 n); overall summaries pool the equally repeated calls of one method. These are warm,
local, serial microbenchmark observations: repeated calls of the same query are not independent samples and are not
evidence of statistical significance. No latency target is imposed.


## View a saved run

```
uv run --no-sync python -m retrieval_eval.viewer runs/demo/run.json            # serves http://127.0.0.1:8765/ (Ctrl+C stops it)
uv run --no-sync python -m retrieval_eval.viewer runs/demo/run.json --port 0   # any free port; the URL is printed
uv run --no-sync python -m retrieval_eval.viewer runs/demo/run.json --export-html page.html --query q02 --k 5
```

The viewer opens **one completed `run.json`** and needs nothing else: document and query texts and all judgments are embedded
in the run, so a copied file works with no dataset directory, no model cache and no retrieval code (`scripts/viewer_smoke.py`
proves this with those imports blocked). It is standard library only (server-rendered HTML, no JavaScript, no new
dependency), read-only, binds to `127.0.0.1` only, checks the `Host` header, escapes every dynamic string and sends a
restrictive Content-Security-Policy. It refuses to start on a port that already has a listener (on Windows it binds
exclusively; elsewhere the operating system already prevents sharing) and exits 2 with a hint. On narrow screens the method columns stack and
each table scrolls inside its own region, so the page itself does not scroll sideways. It is not intended to be exposed beyond the local machine.

The page shows the run ID and dataset designation; aggregate Recall@k and MRR@k per method with scored, total and
no-judged-relevant counts; median and p95 latency with units, sample counts and the warm CPU conditions; and a query view.
Choose any query and cutoff to see both methods side by side: ranked document IDs, text, ranks, score (labeled by method; the
scales are different and not comparable), relevance label (`relevant`, `nonrelevant`, `unjudged`) and the per-query metrics.
Judged-relevant documents are listed per method as found at rank *r*, missed at this cutoff (with the rank in the saved top-N
if it is there), or not in the saved top-N. A scored query with zero recall is shown as a miss, distinct from a query with no
judged-relevant document, which is shown as unscored (null metrics, not zero).

A run is refused with a readable message (exit 2, nothing served) if it is not valid JSON, is an incomplete run
(`run_failed.json`), is not `completed`, has an unsupported `schema_version` (supported: 1.0, 1.1), is structurally malformed
(the offending field is named), or fails the integrity checks of `export.verify_saved_run` (every metric is recomputed from the
saved rankings and judgments; each saved rank must equal its one-based position in the list).

## Continuous integration

`.github/workflows/tests.yml` runs on every push and pull request on **Ubuntu and Windows**: it installs the locked base environment
(`uv sync --locked`, uv 0.12.17), runs the default suite (`uv run --no-sync pytest -q`, which never downloads the model) and the viewer
smoke check on `examples/demo/run.json`. Each job has a 10-minute timeout, so a hung test fails the job instead of blocking it. Actions are
pinned to commit SHAs. The pinned CPU model smoke check is a separate manual command (table above), not part of CI. The workflow has been
parsed and linted locally; **no hosted run exists yet**, so there is no hosted CI status to report.

## Adapter interface and settings

```python
retriever = BM25Retriever(documents)                                   # or
retriever = EmbeddingRetriever(documents, SentenceTransformerEncoder())
retriever.search(query_text, k) -> list[(doc_id, score)]               # best first, min(k, corpus) items
retriever.describe() -> dict                                            # settings, for the later run manifest
```

| | BM25 | Embedding |
|---|---|---|
| Library | rank-bm25 `BM25Okapi` | sentence-transformers |
| Text used | raw document text | raw document text (model's own tokenizer) |
| Tokenizer | `re.findall(r"[a-z0-9]+", text.casefold())` for documents and queries | model tokenizer; inputs over `max_seq_length` are truncated by the library |
| Parameters | `k1=1.5`, `b=0.75`, `epsilon=0.25`; no stemming, no stop-word removal | `sentence-transformers/all-MiniLM-L6-v2` at revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, CPU, `max_seq_length=256`, dimension 384, document encode batch size 32 |
| Scoring | library BM25 score | adapter L2-normalizes document and query vectors (float64), exhaustive dot product = cosine |
| Built once | token index | document embedding matrix |
| Per search | tokenize query, score all documents | encode query (no cache), score all documents |

Shared behavior: scores are plain finite Python floats; ordering is score descending then `doc_id` ascending
(case-sensitive) over the **full** ranking before truncation; `min(k, corpus_size)` unique known IDs are returned (no
padding, zero/negative scores kept); invalid `k` and blank/non-string queries raise; adapters never receive relevance
judgments; a failing encoder raises, with no fallback to another method or vectors. Scores from different methods are not
comparable.

## Known limitations

- BM25 tokenization is English/ASCII only by design: non-ASCII letters are separators (`"café"` -> `["caf"]`).
- With no stop-word removal, function words can dominate BM25 ranking when a query shares no content word with a document.
- rank-bm25 replaces a negative idf (term in more than half the documents) with `epsilon * average_idf`, which can itself
  be negative on small corpora, so matching documents can score below non-matching ones. Behavior of the library, not
  altered here (pinned by a test).
- A term repeated in a query counts once per occurrence (library behavior).
- The synthetic fixture (`data/synthetic/`) is a developer fixture, not an independent benchmark.
- The synthetic results are from seven scored queries; they do not generalize, and latency numbers depend heavily on the machine.
- `.retrieval_eval.lock` marks an output directory in use; a lock left by a killed process must be deleted by hand.
- The model-cache check compares file sizes, not content hashes.
- On a 390 px-wide screen the ranking tables scroll sideways inside their own region, so the document-text column is off-screen until you scroll it.

## Provenance of checksums

See `data/synthetic/CHECKSUMS.md`: `FIXTURE.sha256` is authoritative; the legacy `SHA256SUMS` is unexplained evidence that
does not match the fixture.
