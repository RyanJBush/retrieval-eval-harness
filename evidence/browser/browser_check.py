"""Evidence script (NOT part of the project). Drives a REAL headless Chromium (Playwright) against the local viewer.
Usage: python browser_check.py <label> <run.json> <port> <outdir>   (PYTHONPATH selects which viewer code is served)"""
import json, subprocess, sys, time, urllib.request
from playwright.sync_api import sync_playwright
label, run_json, port, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
py = "/home/claude/retrieval-eval/.venv/bin/python"
srv = subprocess.Popen([py, "-m", "retrieval_eval.viewer", run_json, "--port", str(port)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
base = f"http://127.0.0.1:{port}"
for _ in range(60):
    try: urllib.request.urlopen(base + "/", timeout=1).read(); break
    except Exception: time.sleep(0.25)
results = {}
try:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for w, h, name in ((1036, 791, "desktop"), (390, 844, "narrow")):
            page = browser.new_context(viewport={"width": w, "height": h}).new_page()
            console = []
            page.on("console", lambda m, c=console: c.append((m.type, m.text)))
            page.on("pageerror", lambda e, c=console: c.append(("pageerror", str(e))))
            page.goto(base + "/")
            page.select_option("select[name=q]", "q02"); page.select_option("select[name=k]", "3")
            page.click("button[type=submit]"); page.wait_for_load_state()
            m = page.evaluate("""() => ({
                url: location.search, innerWidth: innerWidth,
                docScrollWidth: document.documentElement.scrollWidth, docClientWidth: document.documentElement.clientWidth,
                bodyScrollWidth: document.body.scrollWidth,
                tables: document.querySelectorAll('table').length,
                scrollRegions: [...document.querySelectorAll('.table-scroll')].map(e => ({scrollWidth: e.scrollWidth, clientWidth: e.clientWidth, scrollsLocally: e.scrollWidth > e.clientWidth})),
                overflowingTables: [...document.querySelectorAll('table')].filter(t => t.getBoundingClientRect().right > innerWidth + 1).length,
                columns: getComputedStyle(document.querySelector('.cols')).gridTemplateColumns.split(' ').length,
                headers: [...document.querySelectorAll('.col h3')].map(e => e.textContent),
                recallLines: [...document.querySelectorAll('.col > p')].filter(e => e.textContent.startsWith('Recall@')).map(e => e.textContent.slice(0, 40)),
                notes: [...document.querySelectorAll('.col .note')].map(e => e.textContent.slice(0, 70)),
                relevantItems: [...document.querySelectorAll('.col li')].map(e => e.textContent.slice(0, 48)),
                firstEmbeddingRow: (document.querySelectorAll('.col')[1].querySelector('tr:nth-child(2)') || {}).textContent
            })""")
            m["consoleMessages"] = console
            m["horizontalPageScroll"] = m["docScrollWidth"] > m["docClientWidth"]
            page.screenshot(path=f"{out}/{label}_{name}_q02_k3.png", full_page=True)
            results[name] = m
        browser.close()
finally:
    srv.terminate(); srv.wait(timeout=10)
json.dump(results, open(f"{out}/{label}_metrics.json", "w"), indent=1)
for name, m in results.items():
    print(f"[{label}] {name} {m['innerWidth']}px: page scrollWidth={m['docScrollWidth']} clientWidth={m['docClientWidth']} horizontal page scroll={m['horizontalPageScroll']} | tables={m['tables']} overflowing={m['overflowingTables']} | scroll regions={len(m['scrollRegions'])} (scrolling locally: {sum(r['scrollsLocally'] for r in m['scrollRegions'])}) | result columns={m['columns']} | console msgs={len(m['consoleMessages'])}")
    print("      ", m["recallLines"], "|", m["relevantItems"])
