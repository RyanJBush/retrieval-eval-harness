Clean copy of the final package: /tmp/walk/retrieval-eval-harness (61 files, no .venv)

$ uv sync --locked
 + packaging==26.3
 + pluggy==1.7.0
 + pygments==2.21.0
 + pytest==9.2.0
 + rank-bm25==0.2.2
 + retrieval-eval==0.1.0 (from file:///tmp/walk/retrieval-eval-harness)
[exit code: 0]

=== README quick start + demo: viewer started normally, pages fetched, second viewer refused, Ctrl+C ===
viewing run a2a9d02d2f2b4698a07a02c470e68c18 (synthetic) at http://127.0.0.1:36809/  (Ctrl-C to stop)
GET / -> HTTP 200
   ok   Dataset designation: synthetic
   ok   0.7143
   ok   0.8571
   ok   n (measured searches)
   ok   median (ms)
GET /?q=q02&k=3 -> HTTP 200
   ok   Scored, zero recall
   ok   d10: missed: not in the saved top-5
   ok   d10: found at rank 1
GET /?q=q03&k=5 -> HTTP 200
   ok   d02: missed: not in the saved top-5
   ok   d02: found at rank 2
   ok   Recall@5: <strong>0.5000</strong>
GET /?q=q03&k=1 -> HTTP 200
   ok   d02: missed at k=1: it is at rank 2 of the saved top-5
GET /?q=q07&k=5 -> HTTP 200
   ok   d19: missed: not in the saved top-5
   ok   d19: found at rank 2
GET /?q=q08&k=3 -> HTTP 200
   ok   Unscored: this query has no judged-relevant document
   ok   Recall@3: <strong>n/a</strong>
GET /?q=q01&k=1 -> HTTP 200
   ok   d01: found at rank 1
   ok   E4107
second viewer on the same port: exit code 2 | error: cannot start the viewer on 127.0.0.1:36809: [Errno 98] Address already in use (the port may already be in use; choose another with --port N, or use --port 0 for any free port)
Ctrl+C (SIGINT): viewer exit code 0 | stderr ''
port closed after stop: URLError
README demo page checks failed: 0

=== README commands table, base environment ===

$ uv run --no-sync pytest -q
........................................................................ [ 89%]
...........................................                              [100%]
403 passed, 12 deselected in 11.72s
[exit code: 0]

$ uv run --no-sync python scripts/viewer_smoke.py examples/demo/run.json --expect-fixture-demo
PASS no model cache was created or consulted

14/14 checks passed (blocked imports: torch, sentence_transformers, transformers, rank_bm25, numpy, huggingface_hub, retrieval_eval.retrievers; working directory /tmp/viewer_smoke_hwxl6mtu)
[exit code: 0]

=== README: model environment ===

$ uv sync --locked --extra model
 + typer==0.28.0
 + typing-extensions==4.16.0
[exit code: 0]

$ uv run --no-sync pytest -m model -q
............                                                             [100%]
12 passed, 403 deselected in 12.52s
[exit code: 0]

$ uv run --no-sync pytest -m 'model or not model' -q
.......................................................                  [100%]
415 passed in 26.25s
[exit code: 0]

$ uv run --no-sync python scripts/smoke_model.py

11/11 checks passed
[exit code: 0]

=== README: run an evaluation, then view it ===

$ uv run --no-sync python -m retrieval_eval.cli --dataset data/synthetic --out-dir runs/demo
status: completed   run_id: a5975749c8d94e36a9394d251d414318
dataset: synthetic (20 documents, 8 queries: 7 scored, 1 with no judged-relevant document)
quality (macro mean over the 7 scored queries; synthetic demo observations):
  method      k   Recall      MRR
  bm25        1   0.7143   0.8571
  bm25        3   0.7143   0.8571
  bm25        5   0.7143   0.8571
  embedding   1   0.8571   1.0000
  embedding   3   1.0000   1.0000
  embedding   5   1.0000   1.0000
latency (warm, local, serial microbenchmark; pooled over all queries; milliseconds):
  bm25       n=160  median=0.095 ms  p95=0.163 ms
  embedding  n=160  median=10.791 ms  p95=14.022 ms
wrote 6 files to runs/demo: summary.csv, per_query.csv, rankings.csv, latency.csv, latency_summary.csv, run.json
[exit code: 0]
latency.csv
latency_summary.csv
per_query.csv
rankings.csv
run.json
summary.csv
fresh run vs bundled example: identity_equal True (differences: [] ) | rankings_equal True | quality_equal True | max score diff 0.0
fresh run verify_saved_run: []

$ uv run --no-sync python scripts/viewer_smoke.py runs/demo/run.json --expect-fixture-demo

14/14 checks passed (blocked imports: torch, sentence_transformers, transformers, rank_bm25, numpy, huggingface_hub, retrieval_eval.retrievers; working directory /tmp/viewer_smoke_a9t9ap3u)
[exit code: 0]

fixture after everything: documents.jsonl: OK queries.jsonl: OK qrels.csv: OK 

README demo check exit status: 0
