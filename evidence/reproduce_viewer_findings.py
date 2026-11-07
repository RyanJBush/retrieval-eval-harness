"""Evidence script (NOT part of the project). Reproduces handoff-06 findings 2 and 3 on whichever viewer code PYTHONPATH selects.
Finding 2: a valid run with only the rank of the first embedding result for q02 changed 1 -> 99.
Finding 3: the file {"status":"incomplete","errors":{"stage":"build"}}."""
import copy, json, subprocess, sys, tempfile
from pathlib import Path
import retrieval_eval
from retrieval_eval import viewer
print("viewer code:", viewer.__file__)
run = json.load(open(sys.argv[1])); tmp = Path(tempfile.mkdtemp())
bad = copy.deepcopy(run); q = bad["results"]["embedding"]["queries"][1]; assert q["query_id"] == "q02"
q["ranking"][0]["rank"] = 99
(tmp / "wrong_rank.json").write_text(json.dumps(bad)); (tmp / "run_failed.json").write_text('{"status":"incomplete","errors":{"stage":"build"}}')
print("\n--- finding 2: rank 1 -> 99 for the first embedding result of q02")
try:
    r = viewer.load_for_view(tmp / "wrong_rank.json"); v = viewer.build_query_view(r, "q02", 5); m = next(x for x in v["methods"] if x["method"] == "embedding")
    print("ACCEPTED. Viewer shows Recall@5 =", m["recall"], "and", [d["detail"] for d in m["relevant_documents"]], "<- contradiction")
except viewer.ViewerError as exc:
    print("REJECTED with a readable message:", str(exc)[:160])
print("\n--- finding 3: malformed failure manifest through the command line")
p = subprocess.run([sys.executable, "-m", "retrieval_eval.viewer", str(tmp / "run_failed.json")], capture_output=True, text=True, timeout=30)
print("exit code", p.returncode); print("stderr:", (p.stderr.strip().splitlines() or [""])[-1][:200]); print("traceback shown:", "Traceback" in p.stderr)
