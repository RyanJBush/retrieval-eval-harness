"""Standalone, read-only viewer for ONE completed saved run (``run.json``).

    python -m retrieval_eval.viewer path/to/run.json [--port 8765]
    python -m retrieval_eval.viewer path/to/run.json --export-html page.html [--query q02] [--k 3]

Design
  * Everything shown comes from the single authoritative JSON run: dataset designation, document and query texts, judgments,
    rankings, metrics, latencies. No dataset directory, model, cache or retrieval code is needed, so a copied artifact works.
  * Display logic only: this module neither retrieves nor evaluates. It reuses the saved-run loader and integrity checks
    from ``export`` (which recompute every metric from the saved rankings) and shows a readable message if they reject a run.
  * Standard library only; server-side rendered HTML, no JavaScript. The server is read-only (GET), binds to 127.0.0.1 only,
    checks the Host header, escapes every dynamic string, and sends a restrictive Content-Security-Policy.
"""

from __future__ import annotations

import argparse
import html
import http.server
import json
import socket
import sys
import threading
import urllib.parse
from pathlib import Path

from . import export
from .runner import SUPPORTED_SCHEMA_VERSIONS

DEFAULT_PORT = 8765
HOST = "127.0.0.1"
MAX_PROBLEMS_SHOWN = 5

SCORE_SCALES = {
    "bm25": "BM25 score, unbounded; larger is better",
    "embedding": "cosine similarity, between -1 and 1; larger is better",
}
DEFAULT_SCORE_SCALE = "method-specific score"


class ViewerError(Exception):
    """A run cannot be shown. The message is meant for a person, not a stack trace."""


# ------------------------------------------------------------------------------------------------- loading and validation
def _shape(run, spec: dict, where: str) -> None:
    """Minimal structural check: every key in ``spec`` exists and has the given type; nested dict specs recurse."""
    for key, want in spec.items():
        path = f"{where}.{key}" if where else key
        if not isinstance(run, dict) or key not in run:
            raise ViewerError(f"malformed run: '{path}' is missing")
        value = run[key]
        if isinstance(want, dict):
            if not isinstance(value, dict):
                raise ViewerError(f"malformed run: '{path}' should be an object")
            _shape(value, want, path)
        elif not isinstance(value, want):
            names = want.__name__ if isinstance(want, type) else "/".join(t.__name__ for t in want)
            raise ViewerError(f"malformed run: '{path}' should be {names}, got {type(value).__name__}")


def check_structure(run: dict) -> None:
    _shape(run, {
        "run": {"run_id": str, "status": str, "started_utc": str, "finished_utc": str},
        "config": {"methods": list, "ks": list, "max_k": int, "warmup_per_query_method": int,
                   "measured_per_query_method": int, "threads_requested": int, "device": str},
        "dataset": {"designation": str, "counts": dict, "documents": list, "queries": list, "judgments": list},
        "results": dict,
        "latency": {"unit": str, "samples": list, "summary": dict},
    }, "")
    for m in run["config"]["methods"]:
        if m not in run["results"]:
            raise ViewerError(f"malformed run: 'results.{m}' is missing")
        _shape(run["results"][m], {"counts": {"total": int, "scored": int, "no_judged_relevant": int},
                                   "aggregate": dict, "queries": list}, f"results.{m}")
        for q in run["results"][m]["queries"]:
            _shape(q, {"query_id": str, "status": str, "n_relevant": int, "n_returned": int, "metrics": dict,
                       "ranking": list}, f"results.{m}.queries[]")
            for r in q["ranking"]:
                _shape(r, {"rank": int, "doc_id": str, "score": (int, float), "label": str}, f"results.{m}.ranking[]")
        _shape(run["latency"]["summary"], {m: {"overall": {"n": int, "median_ns": (int, float), "p95_ns": (int, float)},
                                               "per_query": dict}}, "latency.summary")
        for q in run["results"][m]["queries"]:
            _shape(q, {"measured_search_consistency": {"n_measured": int, "n_ranking_identical": int}},
                   f"results.{m}.queries[]")
    for s in run["latency"]["samples"]:
        _shape(s, {"method": str, "query_id": str, "repetition": int, "execution_index": int, "elapsed_ns": int},
               "latency.samples[]")
    for j in run["dataset"]["judgments"]:
        if not (isinstance(j, list) and len(j) == 3):
            raise ViewerError("malformed run: 'dataset.judgments[]' entries should be [query_id, doc_id, relevance]")
    for d in run["dataset"]["documents"]:
        _shape(d, {"doc_id": str, "text": str}, "dataset.documents[]")
    for q in run["dataset"]["queries"]:
        _shape(q, {"query_id": str, "text": str}, "dataset.queries[]")
    if not run["dataset"]["queries"] or not run["config"]["ks"] or not run["config"]["methods"]:
        raise ViewerError("malformed run: it has no queries, cutoffs or methods")


def load_for_view(path) -> dict:
    """Load and validate a saved run for display, or raise ViewerError with a readable message."""
    p = Path(path)
    if not p.is_file():
        raise ViewerError(f"file not found: {p}")
    try:
        run = export.load_run(p)  # rejects NaN/Infinity constants
    except UnicodeDecodeError:
        raise ViewerError(f"{p.name} is not valid UTF-8 text; expected a run.json written by retrieval_eval.cli") from None
    except json.JSONDecodeError as exc:
        raise ViewerError(f"{p.name} is not valid JSON ({exc.msg} at line {exc.lineno}, column {exc.colno})") from None
    except ValueError as exc:
        raise ViewerError(f"{p.name} is not a valid saved run: {exc}") from None
    except OSError as exc:
        raise ViewerError(f"cannot read {p}: {exc}") from None
    if not isinstance(run, dict):
        raise ViewerError(f"{p.name} is not a saved run: expected a JSON object at the top level")
    if run.get("status") == "incomplete":  # run_failed.json
        errs = run.get("errors")
        e = errs[0] if isinstance(errs, list) and errs and isinstance(errs[0], dict) else {}
        where = ", ".join(f"{k}={e[k]}" for k in ("stage", "method", "query_id") if e.get(k))
        raise ViewerError(f"this is an INCOMPLETE run (a failure manifest, not results)"
                          + (f": {where}" if where else "") + (f": {e.get('error_type', '')}: {e.get('message', '')}"
                                                                if e.get("message") else "")
                          + ". Only a completed run.json can be viewed.")
    if "run" in run and isinstance(run["run"], dict) and run["run"].get("status") != "completed":
        raise ViewerError(f"run status is {run['run'].get('status')!r}, not 'completed'; only completed runs can be viewed")
    version = run.get("schema_version")
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ViewerError(f"unsupported run schema_version {version!r}; this viewer reads: "
                          f"{', '.join(SUPPORTED_SCHEMA_VERSIONS)}")
    check_structure(run)
    try:
        problems = export.verify_saved_run(run)
    except Exception as exc:  # noqa: BLE001 - a run so malformed that the checks themselves cannot run
        detail = (f"a required field is missing: {exc.args[0]!r}" if isinstance(exc, KeyError) and exc.args
                  else f"unexpected data ({exc})")
        raise ViewerError(f"malformed run: the integrity checks could not read the saved data: {detail}") from None
    if problems:
        shown = "; ".join(problems[:MAX_PROBLEMS_SHOWN]) + (f"; ... and {len(problems) - MAX_PROBLEMS_SHOWN} more"
                                                            if len(problems) > MAX_PROBLEMS_SHOWN else "")
        raise ViewerError(f"the saved run failed its integrity checks and is not shown: {shown}")
    return run


# ------------------------------------------------------------------------------------------------- display model (no HTML)
def _fmt(value, digits=4):
    return None if value is None else f"{value:.{digits}f}"


def score_scale(method: str) -> str:
    return SCORE_SCALES.get(method, DEFAULT_SCORE_SCALE)


def build_overview(run: dict) -> dict:
    cfg, ds, env = run["config"], run["dataset"], run.get("environment") or {}
    tp = run.get("timing_protocol") or {}
    rows = []
    for m in cfg["methods"]:
        res = run["results"][m]
        for k in sorted(cfg["ks"]):
            a = res["aggregate"][str(k)]
            rows.append({"method": m, "k": k, "macro_recall": a["macro_recall"], "mrr": a["mrr"],
                         "scored": res["counts"]["scored"], "total": res["counts"]["total"],
                         "no_judged_relevant": res["counts"]["no_judged_relevant"]})
    latency = {}
    for m in cfg["methods"]:
        o = run["latency"]["summary"][m]["overall"]
        latency[m] = {"n": o["n"], "median_ns": o["median_ns"], "p95_ns": o["p95_ns"],
                      "median_ms": o["median_ns"] / 1e6, "p95_ms": o["p95_ns"] / 1e6}
    synthetic = ds["designation"] == "synthetic"
    model = run.get("model") or {}
    cpu = env.get("cpu_model", "unknown")
    return {
        "run_id": run["run"]["run_id"], "status": run["run"]["status"], "schema_version": run["schema_version"],
        "started_utc": run["run"]["started_utc"], "finished_utc": run["run"]["finished_utc"],
        "designation": ds["designation"],
        "designation_note": ("Synthetic developer fixture: these results demonstrate that the pipeline works; they are not "
                             "a benchmark and say nothing about retrieval quality on real data."
                             if synthetic else "External dataset: see the dataset provenance in the run manifest."),
        "counts": ds["counts"], "methods": list(cfg["methods"]), "ks": sorted(cfg["ks"]), "max_k": cfg["max_k"],
        "model": (f"{model['model_id']} @ {str(model.get('revision', ''))[:12]}" if model.get("model_id") else None),
        "aggregate_rows": rows, "latency": latency, "latency_unit_note": "milliseconds (ms), from integer-nanosecond samples",
        "conditions": (f"Warm, local, serial {cfg['device'].upper()} microbenchmark. Per query and method: "
                       f"{cfg['warmup_per_query_method']} discarded warm-up searches, then "
                       f"{cfg['measured_per_query_method']} measured searches (alternating which method runs first); "
                       f"{cfg['threads_requested']} thread(s); no caches. Machine: {env.get('platform', 'unknown')}; "
                       f"CPU: {cpu}; Python {env.get('python_version', 'unknown')}. n counts all measured searches of "
                       "the method, pooled over queries. Repeated searches are not independent samples, so no "
                       "significance is implied; timings vary between runs."),
        "timed_region": tp.get("timed_region"),
        "consistency": run.get("ranking_consistency"),
    }


def build_query_view(run: dict, query_id: str, k: int) -> dict:
    cfg = run["config"]
    texts = {d["doc_id"]: d["text"] for d in run["dataset"]["documents"]}
    queries = {q["query_id"]: q["text"] for q in run["dataset"]["queries"]}
    if query_id not in queries:
        raise ViewerError(f"unknown query {query_id!r}")
    if k not in cfg["ks"]:
        raise ViewerError(f"cutoff k={k} is not in this run (available: {', '.join(map(str, sorted(cfg['ks'])))})")
    relevant = sorted(d for q, d, r in run["dataset"]["judgments"] if q == query_id and r == 1)
    methods = []
    for m in cfg["methods"]:
        rec = next(q for q in run["results"][m]["queries"] if q["query_id"] == query_id)
        met = rec["metrics"][str(k)]
        ranking = rec["ranking"]
        rank_of = {r["doc_id"]: r["rank"] for r in ranking}
        if rec["status"] == "no_judged_relevant":
            verdict = "unscored"
            note = "Unscored: this query has no judged-relevant document, so Recall and RR are null (not zero)."
        elif met["recall"] == 0.0:
            verdict = "zero_recall"
            note = (f"Scored, zero recall: the query has {rec['n_relevant']} judged-relevant document(s) and none is in the "
                    f"top {k}. This is a miss, not an unscored query.")
        else:
            verdict = "scored"
            note = None
        misses = []
        for d in relevant:
            r = rank_of.get(d)
            if r is not None and r <= k:
                outcome, detail = "found", f"found at rank {r}"
            elif r is not None:
                outcome, detail = "missed", f"missed at k={k}: it is at rank {r} of the saved top-{len(ranking)}"
            else:
                outcome, detail = "missed", f"missed: not in the saved top-{len(ranking)}"
            misses.append({"doc_id": d, "outcome": outcome, "detail": detail, "text": texts.get(d, "")})
        methods.append({
            "method": m, "score_scale": score_scale(m), "status": rec["status"], "verdict": verdict, "note": note,
            "recall": met["recall"], "rr": met["rr"], "n_relevant": rec["n_relevant"], "n_returned": rec["n_returned"],
            "rows": [{"rank": r["rank"], "doc_id": r["doc_id"], "score": r["score"], "label": r["label"],
                      "text": texts.get(r["doc_id"], "")} for r in ranking if r["rank"] <= k],
            "relevant_documents": misses,
        })
    return {"query_id": query_id, "text": queries[query_id], "k": k, "relevant_ids": relevant, "methods": methods}


# ------------------------------------------------------------------------------------------------- HTML rendering
def esc(value) -> str:
    return html.escape(str(value), quote=True)


STYLE = """
body{font-family:system-ui,sans-serif;margin:1.5rem;max-width:78rem;color:#1b1b1b;overflow-wrap:anywhere}
h1{font-size:1.4rem} h2{font-size:1.1rem;margin-top:1.8rem;border-bottom:1px solid #ccc;padding-bottom:.2rem}
table{border-collapse:collapse;margin:.4rem 0} th,td{border:1px solid #bbb;padding:.25rem .5rem;text-align:left;vertical-align:top}
.table-scroll{max-width:100%;overflow-x:auto} .table-scroll table{overflow-wrap:normal}
select{max-width:100%}
th{background:#f0f0f0} td.num{text-align:right;font-variant-numeric:tabular-nums}
.note{background:#fff7d6;border:1px solid #e0c96a;padding:.5rem .7rem;margin:.6rem 0}
.error{background:#ffe3e3;border:1px solid #d06a6a;padding:.5rem .7rem;margin:.6rem 0}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:1rem} .col{min-width:0}
.badge{font-weight:600;padding:0 .35rem;border:1px solid #888;border-radius:.3rem}
.relevant{background:#d9f2d9}.nonrelevant{background:#f4f4f4}.unjudged{background:#e7e7ff}
.found{color:#145214}.missed{color:#8a1010;font-weight:600} .muted{color:#555;font-size:.9rem}
.verdict-zero_recall{border-left:.4rem solid #b00020;padding-left:.5rem}.verdict-unscored{border-left:.4rem solid #777;padding-left:.5rem}
@media (max-width:60rem){.cols{grid-template-columns:1fr}}
"""


def _td(value, cls="") -> str:
    return f'<td class="{cls}">{esc(value)}</td>' if cls else f"<td>{esc(value)}</td>"


def _metric(v, digits=4) -> str:
    return "n/a" if v is None else f"{v:.{digits}f}"


def _render_overview(ov: dict) -> str:
    h = [f"<h1>Saved retrieval run <code>{esc(ov['run_id'])}</code></h1>",
         f'<p><strong>Dataset designation: {esc(ov["designation"])}</strong> &middot; status: {esc(ov["status"])} &middot; '
         f"schema {esc(ov['schema_version'])} &middot; started {esc(ov['started_utc'])} &middot; finished {esc(ov['finished_utc'])}</p>",
         f'<div class="note">{esc(ov["designation_note"])}</div>',
         f"<p>{esc(ov['counts']['documents'])} documents, {esc(ov['counts']['queries'])} queries, "
         f"{esc(ov['counts']['judgments'])} judgments ({esc(ov['counts']['positive_judgments'])} relevant)"
         + (f"; embedding model {esc(ov['model'])}" if ov["model"] else "") + "</p>"]
    if ov.get("consistency"):
        c = ov["consistency"]
        h.append(f'<p class="muted">Repeat-search consistency: {esc(c.get("ranking_disagreements"))} ranking disagreements and '
                 f'{esc(c.get("score_only_differences"))} score-only differences in {esc(c.get("calls_checked"))} checked searches.</p>')
    h.append('<h2>Quality by method and cutoff</h2><div class="table-scroll" role="region" aria-label="Quality results" tabindex="0">'
             "<table><tr><th>method</th><th>k</th><th>macro Recall@k</th>"
             "<th>MRR@k</th><th>scored queries</th><th>total queries</th><th>no judged-relevant</th></tr>")
    for r in ov["aggregate_rows"]:
        h.append("<tr>" + _td(r["method"]) + _td(r["k"], "num") + _td(_metric(r["macro_recall"]), "num")
                 + _td(_metric(r["mrr"]), "num") + _td(r["scored"], "num") + _td(r["total"], "num")
                 + _td(r["no_judged_relevant"], "num") + "</tr>")
    h.append("</table></div><p class='muted'>Macro means are over scored queries only; a query with no judged-relevant document is "
             "unscored (null), not zero. n/a means no query could be scored.</p>")
    h.append(f"<h2>Retrieval latency ({esc(ov['latency_unit_note'])})</h2>"
             '<div class="table-scroll" role="region" aria-label="Latency results" tabindex="0">'
             "<table><tr><th>method</th><th>n (measured searches)</th>"
             "<th>median (ms)</th><th>p95 (ms)</th></tr>")
    for m, v in ov["latency"].items():
        h.append("<tr>" + _td(m) + _td(v["n"], "num") + _td(f"{v['median_ms']:.3f}", "num") + _td(f"{v['p95_ms']:.3f}", "num") + "</tr>")
    h.append(f"</table></div><p class='muted'>{esc(ov['conditions'])} p95 is the nearest-rank value at rank ceil(0.95 n).</p>")
    if ov.get("timed_region"):
        h.append(f"<p class='muted'>Timed region: {esc(ov['timed_region'])}</p>")
    return "\n".join(h)


def _render_query(run: dict, ov: dict, view: dict) -> str:
    h = ["<h2>Query view</h2>",
         '<form method="get" action="/"><label>Query <select name="q">']
    for q in run["dataset"]["queries"]:
        sel = " selected" if q["query_id"] == view["query_id"] else ""
        h.append(f'<option value="{esc(q["query_id"])}"{sel}>{esc(q["query_id"])}: {esc(q["text"])}</option>')
    h.append('</select></label> <label>Cutoff k <select name="k">')
    for k in ov["ks"]:
        sel = " selected" if k == view["k"] else ""
        h.append(f'<option value="{esc(k)}"{sel}>{esc(k)}</option>')
    h.append('</select></label> <button type="submit">Show</button></form>')
    h.append(f"<p><strong>{esc(view['query_id'])}</strong>: <em>{esc(view['text'])}</em> &middot; cutoff k = {esc(view['k'])}</p>")
    h.append('<div class="cols">')
    for m in view["methods"]:
        h.append(f'<div class="col verdict-{esc(m["verdict"])}"><h3>{esc(m["method"])}</h3>')
        h.append(f"<p>Recall@{esc(view['k'])}: <strong>{esc(_metric(m['recall']))}</strong> &middot; "
                 f"RR@{esc(view['k'])}: <strong>{esc(_metric(m['rr']))}</strong> &middot; status: {esc(m['status'])} &middot; "
                 f"{esc(m['n_relevant'])} judged relevant &middot; {esc(m['n_returned'])} saved results</p>")
        if m["note"]:
            h.append(f'<div class="note">{esc(m["note"])}</div>')
        h.append(f'<div class="table-scroll" role="region" aria-label="{esc(m["method"])} rankings" tabindex="0">'
                 f"<table><tr><th>rank</th><th>doc</th><th>score ({esc(m['method'])}: {esc(m['score_scale'])})</th>"
                 "<th>label</th><th>text</th></tr>")
        if not m["rows"]:
            h.append('<tr><td colspan="5">no results returned for this query</td></tr>')
        for r in m["rows"]:
            h.append("<tr>" + _td(r["rank"], "num") + _td(r["doc_id"]) + _td(f"{r['score']:.6g}", "num")
                     + f'<td><span class="badge {esc(r["label"])}">{esc(r["label"])}</span></td>' + _td(r["text"]) + "</tr>")
        h.append("</table></div>")
        h.append("<p><strong>Judged-relevant documents</strong></p>")
        if not m["relevant_documents"]:
            h.append("<p class='muted'>none (no judged-relevant document for this query)</p>")
        else:
            h.append("<ul>")
            for d in m["relevant_documents"]:
                h.append(f'<li class="{esc(d["outcome"])}">{esc(d["doc_id"])}: {esc(d["detail"])}'
                         f' &mdash; <span class="muted">{esc(d["text"])}</span></li>')
            h.append("</ul>")
        h.append("</div>")
    h.append("</div>")
    h.append("<p class='muted'>Scores are on different scales per method and must not be compared across the two columns. "
             "Labels come from the saved judgments: relevant, nonrelevant (explicit 0), or unjudged (no judgment).</p>")
    return "\n".join(h)


def render_page(run: dict, query_id: str | None = None, k: int | None = None, error: str | None = None) -> str:
    ov = build_overview(run)
    qid = query_id if query_id is not None else run["dataset"]["queries"][0]["query_id"]
    kk = k if k is not None else max(ov["ks"])
    body = [_render_overview(ov)]
    if error:
        body.insert(0, f'<div class="error">{esc(error)}</div>')
        qid, kk = run["dataset"]["queries"][0]["query_id"], max(ov["ks"])
    body.append(_render_query(run, ov, build_query_view(run, qid, kk)))
    return ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"><title>Run "
            f"{esc(ov['run_id'])}</title><style>{STYLE}</style></head><body>" + "\n".join(body) + "</body></html>\n")


# ------------------------------------------------------------------------------------------------- local server
def parse_selection(run: dict, query_string: str):
    """Validate ?q=&k= against the run. Returns (query_id, k) or raises ViewerError. Missing values use the defaults."""
    params = urllib.parse.parse_qs(query_string, keep_blank_values=True)
    ids = [q["query_id"] for q in run["dataset"]["queries"]]
    ks = sorted(run["config"]["ks"])
    qid = params["q"][0] if "q" in params else ids[0]
    if qid not in ids:
        raise ViewerError(f"unknown query {qid!r}; choose one of: {', '.join(ids)}")
    if "k" in params:
        raw = params["k"][0]
        if not (raw.isascii() and raw.isdigit()) or int(raw) not in ks:
            raise ViewerError(f"invalid cutoff {raw!r}; choose one of: {', '.join(map(str, ks))}")
        k = int(raw)
    else:
        k = max(ks)
    return qid, k


# Windows: SO_REUSEADDR lets a second socket bind a port that already has a live listener (POSIX does not), so two viewers
# could silently share one port. There, address reuse is switched off and SO_EXCLUSIVEADDRUSE is set before bind().
_WINDOWS_EXCLUSIVE_FALLBACK = -5  # ~SO_REUSEADDR, the documented value of SO_EXCLUSIVEADDRUSE


def reuse_address_allowed(platform: str = sys.platform) -> bool:
    return platform != "win32"


def apply_exclusive_bind(sock, platform: str = sys.platform) -> None:
    """Make binding exclusive on Windows; a no-op elsewhere (where a live listener already blocks a second bind)."""
    if platform == "win32":
        sock.setsockopt(socket.SOL_SOCKET, getattr(socket, "SO_EXCLUSIVEADDRUSE", _WINDOWS_EXCLUSIVE_FALLBACK), 1)


class _ViewerServer(http.server.ThreadingHTTPServer):
    allow_reuse_address = reuse_address_allowed()

    def server_bind(self) -> None:
        apply_exclusive_bind(self.socket)
        super().server_bind()


def make_server(run: dict, port: int = 0) -> http.server.ThreadingHTTPServer:
    allowed_hosts = set()

    class Handler(http.server.BaseHTTPRequestHandler):
        server_version = "RunViewer"

        def log_message(self, *args):  # keep the console quiet; the page itself is the interface
            pass

        def _send(self, status: int, body: str) -> None:
            data = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'none'; style-src 'unsafe-inline'; form-action 'self'")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):  # noqa: N802
            if (self.headers.get("Host") or "") not in allowed_hosts:  # blocks DNS-rebinding style requests
                return self._send(421, "<!doctype html><p>Misdirected request: use http://127.0.0.1:PORT/ or localhost.</p>")
            parts = urllib.parse.urlsplit(self.path)
            if parts.path != "/":
                return self._send(404, "<!doctype html><p>Not found. The only page is <a href=\"/\">/</a>.</p>")
            try:
                qid, k = parse_selection(run, parts.query)
            except ViewerError as exc:
                return self._send(400, render_page(run, error=str(exc)))
            return self._send(200, render_page(run, qid, k))

        def _method_not_allowed(self):
            self.send_response(405)
            self.send_header("Allow", "GET")
            self.send_header("Content-Length", "0")
            self.end_headers()

        do_POST = do_PUT = do_DELETE = do_PATCH = do_HEAD = do_OPTIONS = _method_not_allowed

    server = _ViewerServer((HOST, port), Handler)
    allowed_hosts.update({f"127.0.0.1:{server.server_port}", f"localhost:{server.server_port}"})
    return server


def serve_in_thread(run: dict, port: int = 0):
    """Start a server on a background thread (tests, smoke checks). Returns (server, base_url)."""
    server = make_server(run, port)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_port}"


# ------------------------------------------------------------------------------------------------- command line
def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m retrieval_eval.viewer",
                                description="View one completed saved run (run.json) in a local read-only web page.")
    p.add_argument("run_json", help="path to a completed run.json written by retrieval_eval.cli")
    p.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"local port on {HOST} (default {DEFAULT_PORT}; 0 = any free port)")
    p.add_argument("--export-html", metavar="FILE", help="write one static page instead of starting a server")
    p.add_argument("--query", help="query id for --export-html (default: the first query)")
    p.add_argument("--k", type=int, help="cutoff for --export-html (default: the largest saved cutoff)")
    try:
        args = p.parse_args(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 2
    try:
        run = load_for_view(args.run_json)
        if args.export_html:
            qid, k = parse_selection(run, urllib.parse.urlencode({key: v for key, v in (("q", args.query), ("k", args.k)) if v is not None}))
            Path(args.export_html).write_text(render_page(run, qid, k), encoding="utf-8")
            print(f"wrote {args.export_html} (run {run['run']['run_id']}, query {qid}, k={k})")
            return 0
        if not 0 <= args.port <= 65535:
            raise ViewerError(f"invalid port {args.port}")
        server = make_server(run, args.port)
    except ViewerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"error: cannot start the viewer on {HOST}:{args.port}: {exc} (the port may already be in use; "
              "choose another with --port N, or use --port 0 for any free port)", file=sys.stderr)
        return 2
    print(
        f"viewing run {run['run']['run_id']} ({run['dataset']['designation']}) "
        f"at http://127.0.0.1:{server.server_port}/  (Ctrl-C to stop)",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
