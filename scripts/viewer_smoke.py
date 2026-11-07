"""Viewer smoke check on a COPIED saved run with retrieval and model code unavailable. Exits nonzero if any check fails.

    uv run --frozen python scripts/viewer_smoke.py path/to/run.json [--expect-fixture-demo]

What it does
  1. Copies run.json into a fresh temporary directory (no dataset, no model cache, different working directory).
  2. Blocks every retrieval/model import (torch, sentence-transformers, transformers, rank-bm25, numpy, huggingface-hub and
     retrieval_eval.retrievers) BEFORE importing the viewer, so any dependency on them fails loudly.
  3. Starts the viewer on a free local port and fetches the page for every (query, cutoff). The rendered rows (rank, doc ID,
     score, label) and the missed-relevant statements are parsed back out of the HTML and compared with the saved JSON and
     judgments, recomputed independently here.
  4. With --expect-fixture-demo (the frozen synthetic fixture): checks the q01, q02, q03, q07 and q08 demo paths.
  5. Checks that an incomplete run and a tampered copy are refused with a readable message.
"""

from __future__ import annotations

import html.parser
import http.client
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

BLOCKED = ["torch", "sentence_transformers", "transformers", "rank_bm25", "numpy", "huggingface_hub",
           "retrieval_eval.retrievers"]
CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(ok), detail))


class ColumnParser(html.parser.HTMLParser):
    """Collects, per method column (<div class="col ...">), the table rows and the 'judged-relevant' list items."""

    def __init__(self):
        super().__init__()
        self.cols: list[dict] = []
        self._in_col = self._in_tr = self._in_td = self._in_li = self._in_h3 = False
        self._cell: list[str] = []
        self._row: list[str] = []
        self._cls = ""
        self._li: list[str] = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "div" and (a.get("class") or "").startswith("col "):
            self.cols.append({"method": None, "rows": [], "relevant": []})
            self._in_col = True
        elif self._in_col and tag == "h3":
            self._in_h3, self._cell = True, []
        elif self._in_col and tag == "tr":
            self._in_tr, self._row = True, []
        elif self._in_col and self._in_tr and tag in ("td", "th"):
            self._in_td, self._cell = True, []
        elif self._in_col and tag == "li":
            self._in_li, self._li = True, []

    def handle_endtag(self, tag):
        if tag == "h3" and self._in_h3:
            self.cols[-1]["method"] = "".join(self._cell)
            self._in_h3 = False
        elif tag in ("td", "th") and self._in_td:
            self._row.append("".join(self._cell).strip())
            self._in_td = False
        elif tag == "tr" and self._in_tr:
            if self._row and self._row[0].isdigit():
                self.cols[-1]["rows"].append(self._row)
            self._in_tr = False
        elif tag == "li" and self._in_li:
            self.cols[-1]["relevant"].append("".join(self._li).strip())
            self._in_li = False

    def handle_data(self, data):
        if self._in_td or self._in_h3:
            self._cell.append(data)
        if self._in_li:
            self._li.append(data)


def fetch(port: int, path: str):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=15)
    conn.request("GET", path)
    r = conn.getresponse()
    body = r.read().decode("utf-8")
    conn.close()
    return r.status, body


def expected_relevance(run, qid, method, k):
    rel = sorted(d for q, d, r in run["dataset"]["judgments"] if q == qid and r == 1)
    ranking = next(q for q in run["results"][method]["queries"] if q["query_id"] == qid)["ranking"]
    pos = {r["doc_id"]: r["rank"] for r in ranking}
    return rel, pos, len(ranking)


def main(argv) -> int:
    args = [a for a in argv if not a.startswith("--")]
    demo = "--expect-fixture-demo" in argv
    if len(args) != 1:
        print(__doc__)
        return 2
    src = Path(args[0])
    work = Path(tempfile.mkdtemp(prefix="viewer_smoke_"))
    start_dir = os.getcwd()
    copied = work / "copied_run.json"
    shutil.copyfile(src, copied)
    os.chdir(work)  # away from the project and any dataset directory
    hf_home = work / "no_model_cache"
    os.environ["HF_HOME"] = str(hf_home)
    for name in BLOCKED:
        sys.modules[name] = None  # any import of these now raises ImportError

    from retrieval_eval import viewer  # noqa: E402

    run = viewer.load_for_view(copied)
    ids = [q["query_id"] for q in run["dataset"]["queries"]]
    ks = sorted(run["config"]["ks"])
    methods = run["config"]["methods"]
    check("copied run loads (schema %s)" % run["schema_version"], True,
          f"run {run['run']['run_id']} designation={run['dataset']['designation']}")

    server, base = viewer.serve_in_thread(run)
    port = server.server_port
    try:
        status, body = fetch(port, "/")
        check("front page shows run id and designation", status == 200 and run["run"]["run_id"] in body
              and f"Dataset designation: {run['dataset']['designation']}" in body)
        ov = viewer.build_overview(run)
        agg_ok = all(f"{r['macro_recall']:.4f}" in body for r in ov["aggregate_rows"] if r["macro_recall"] is not None)
        lat_ok = all(f"{v['median_ms']:.3f}" in body and f"{v['p95_ms']:.3f}" in body for v in ov["latency"].values())
        check("aggregate Recall/MRR and latency (median, p95, ms, n) are shown", agg_ok and lat_ok and "median (ms)" in body
              and "Warm, local, serial" in body)

        mismatches, pages = [], 0
        for qid in ids:
            for k in ks:
                status, page = fetch(port, f"/?q={qid}&k={k}")
                pages += 1
                if status != 200 or f'<option value="{qid}" selected>' not in page:
                    mismatches.append(f"{qid}@{k}: status {status} or selector")
                    continue
                parser = ColumnParser()
                parser.feed(page)
                if [c["method"] for c in parser.cols] != methods:
                    mismatches.append(f"{qid}@{k}: columns {[c['method'] for c in parser.cols]}")
                    continue
                qrels = {d: r for q, d, r in run["dataset"]["judgments"] if q == qid}
                for col, m in zip(parser.cols, methods):
                    saved = next(q for q in run["results"][m]["queries"] if q["query_id"] == qid)["ranking"]
                    want = [[str(r["rank"]), r["doc_id"], f"{r['score']:.6g}",
                             {1: "relevant", 0: "nonrelevant", None: "unjudged"}[qrels.get(r["doc_id"])]]
                            for r in saved if r["rank"] <= k]
                    got = [[r[0], r[1], r[2], r[3]] for r in col["rows"]]
                    if got != want:
                        mismatches.append(f"{qid}@{k}/{m}: rows differ")
                    rel, pos, n_saved = expected_relevance(run, qid, m, k)
                    for d in rel:
                        r = pos.get(d)
                        found = r is not None and r <= k
                        line = next((x for x in col["relevant"] if x.startswith(f"{d}:")), "")
                        if found and f"found at rank {r}" not in line:
                            mismatches.append(f"{qid}@{k}/{m}: {d} should be found at rank {r}")
                        if not found and "missed" not in line:
                            mismatches.append(f"{qid}@{k}/{m}: {d} should be shown as missed")
        check(f"all {pages} pages ({len(ids)} queries x {len(ks)} cutoffs): ranks, doc IDs, scores, labels and "
              "found/missed statements match the saved JSON and judgments", not mismatches,
              "; ".join(mismatches[:5]))

        if demo:
            def col(qid, k, m):
                _, page = fetch(port, f"/?q={qid}&k={k}")
                p = ColumnParser()
                p.feed(page)
                return page, next(c for c in p.cols if c["method"] == m)

            page, bm = col("q02", 5, "bm25")
            _, em = col("q02", 5, "embedding")
            check("q02: BM25 shows zero recall and d10 missed; embedding ranks d10 first and relevant",
                  "Scored, zero recall" in page and any(x.startswith("d10: missed") for x in bm["relevant"])
                  and em["rows"][0][1] == "d10" and em["rows"][0][3] == "relevant"
                  and any(x.startswith("d10: found at rank 1") for x in em["relevant"]))
            for qid, second in (("q03", "d02"), ("q07", "d19")):
                _, bm = col(qid, 5, "bm25")
                _, em = col(qid, 5, "embedding")
                _, em1 = col(qid, 1, "embedding")
                rb = {x.split(":")[0]: x for x in bm["relevant"]}
                re_ = {x.split(":")[0]: x for x in em["relevant"]}
                check(f"{qid}: both find the first relevant document at rank 1; only the embedding finds {second} in the top 5",
                      bm["rows"][0][3] == "relevant" and em["rows"][0][3] == "relevant"
                      and "missed" in rb[second] and "found at rank 2" in re_[second]
                      and any(x.startswith(f"{second}: missed at k=1") for x in em1["relevant"]))
            page8, _ = col("q08", 3, "bm25")
            check("q08: shown as unscored (no judged-relevant document), not as zero recall",
                  "Unscored: this query has no judged-relevant document" in page8 and "Scored, zero recall" not in page8
                  and "Recall@3: <strong>n/a</strong>" in page8)
            _, b1 = col("q01", 1, "bm25")
            _, e1 = col("q01", 1, "embedding")
            check("q01: exact identifier E4107 ranks d01 first for both methods",
                  b1["rows"][0][1] == "d01" == e1["rows"][0][1])

        status, _ = fetch(port, "/?q=nope")
        check("an unknown query is refused with HTTP 400", status == 400)
    finally:
        server.shutdown()
        server.server_close()

    # refusals: an incomplete run and a tampered copy
    failed = work / "run_failed.json"
    failed.write_text(json.dumps({"status": "incomplete", "errors": [{"stage": "build", "method": "embedding",
                                  "error_type": "ConnectionError", "message": "offline"}]}), encoding="utf-8")
    try:
        viewer.load_for_view(failed)
        check("an incomplete run is refused", False)
    except viewer.ViewerError as exc:
        check("an incomplete run is refused with a readable message", "INCOMPLETE" in str(exc), str(exc)[:90])
    tampered = json.loads(copied.read_text(encoding="utf-8"))
    tampered["results"][methods[0]]["queries"][0]["metrics"][str(ks[0])]["recall"] = 0.123
    (work / "tampered.json").write_text(json.dumps(tampered), encoding="utf-8")
    try:
        viewer.load_for_view(work / "tampered.json")
        check("a tampered run is refused", False)
    except viewer.ViewerError as exc:
        check("a tampered run is refused with a readable message", "integrity checks" in str(exc), str(exc)[:90])

    leaked = sorted(m for m in BLOCKED if sys.modules.get(m) is not None)
    check("no retrieval/model module was imported", not leaked, ", ".join(leaked))
    check("no model cache was created or consulted", not hf_home.exists())

    for name, ok, detail in CHECKS:
        print(f"{'PASS' if ok else 'FAIL'} {name}" + (f"  [{detail}]" if detail and (not ok or 'run ' in detail) else ""))
    failed_n = sum(1 for _, ok, _ in CHECKS if not ok)
    print(f"\n{len(CHECKS) - failed_n}/{len(CHECKS)} checks passed (blocked imports: {', '.join(BLOCKED)}; working directory {work})")
    os.chdir(start_dir)  # a directory cannot be removed while it is the working directory (Windows)
    shutil.rmtree(work, ignore_errors=True)
    return 1 if failed_n else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
