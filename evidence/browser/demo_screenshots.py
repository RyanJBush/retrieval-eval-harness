"""Evidence script (NOT part of the project): real headless Chromium screenshots of the README demo paths (desktop 1036x791)."""
import subprocess, sys, time, urllib.request
from playwright.sync_api import sync_playwright
py = "/home/claude/retrieval-eval/.venv/bin/python"; run = sys.argv[1]; port = int(sys.argv[2]); out = sys.argv[3]
srv = subprocess.Popen([py, "-m", "retrieval_eval.viewer", run, "--port", str(port)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
base = f"http://127.0.0.1:{port}"
for _ in range(60):
    try: urllib.request.urlopen(base + "/", timeout=1).read(); break
    except Exception: time.sleep(0.25)
try:
    with sync_playwright() as p:
        b = p.chromium.launch(); page = b.new_context(viewport={"width": 1036, "height": 791}).new_page(); errs = []
        page.on("console", lambda m: errs.append(m.text)); page.on("pageerror", lambda e: errs.append(str(e)))
        for q, k in (("q03", 5), ("q07", 5), ("q08", 3), ("q01", 1)):
            page.goto(base + "/"); page.select_option("select[name=q]", q); page.select_option("select[name=k]", str(k)); page.click("button[type=submit]"); page.wait_for_load_state()
            page.screenshot(path=f"{out}/demo_{q}_k{k}_desktop.png", full_page=True)
            txt = page.inner_text("body")
            keys = {"q03": ["d02: missed", "d02: found at rank 2"], "q07": ["d19: missed", "d19: found at rank 2"], "q08": ["Unscored: this query has no judged-relevant document"], "q01": ["d01: found at rank 1"]}[q]
            print(f"{q}@k={k}: " + "; ".join(f"{'ok' if s in txt else 'MISSING'} '{s}'" for s in keys))
        print("console/page errors across the four pages:", errs)
        b.close()
finally:
    srv.terminate(); srv.wait(timeout=10)
