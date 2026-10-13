"""Click through every combination the demo offers, in a real browser, and check what a visitor would see.

    python frontend/scripts/check_demo_ui.py [--url http://127.0.0.1:8501/] [--records 233 ...]

The options come from the site's own dropdowns (record, mode, stress scenario), so nothing the site offers is
skipped. For each combination it starts a run like a visitor and checks:

  every run       no error box; the summary says who wrote it; the ECG shows a trace
  unreadable      "Not triaged"; the trace is explained; "Need a person" says the whole recording
  complete        summary by recorded Gemma; findings table; "Explain this finding" works and says who explained;
                  Adapter vs text fills in by itself; every suggested question, clicked in turn, gets an answer;
                  the clinician override (first mode of each record and scenario) is recorded
  once            the worklist lists the runs; "How it works" renders

Point it at the live entry point (streamlit_app.py: embedded API, demo presets, uploads off) installed like
Streamlit Community Cloud. Exit code 1 if anything fails.
"""
from __future__ import annotations

import argparse
import re
import sys
import time

from capture_screenshots import pick
from playwright.sync_api import Page, sync_playwright

T = 90_000


def visible(page: Page, text: str, timeout: int = 0) -> bool:
    try:
        page.get_by_text(text).first.wait_for(state="visible", timeout=timeout or 1)
        return True
    except Exception:  # noqa: BLE001 - absence is the answer
        return False


def options(page: Page, label: str) -> list[str]:
    """Every option of the selectbox whose label starts with ``label``, as the site offers it."""
    box = page.locator('[data-testid="stSelectbox"]').filter(has_text=label).first
    box.get_by_role("combobox").click()
    names = [o.inner_text().strip() for o in page.get_by_role("option").all()]
    page.keyboard.press("Escape")
    return names


def errors(page: Page) -> list[str]:
    return ["an error box is shown"] if page.locator('[data-testid="stException"]').count() else []


def check(page: Page, base: str, rec: str, scen: str, mode: str, override: bool) -> tuple[str, list[str]]:
    page.goto(base)
    page.get_by_text("Analyse a recording").wait_for(timeout=T)
    pick(page, "Record", rec)
    pick(page, "Mode", mode)
    pick(page, "Stress scenario", scen)
    page.get_by_role("button", name="Start").click()
    page.wait_for_url("**/run?run=*", timeout=T)
    page.get_by_text("Audit trail").first.wait_for(timeout=T)  # tabs appear once the run has finished
    status = "unreadable" if visible(page, "Not triaged", 3000) else "complete"
    problems = errors(page)
    if visible(page, "written by ?"):
        problems.append('summary says "written by ?"')
    if visible(page, "No signal to show"):
        problems.append("no ECG trace")
    if status == "unreadable":
        if not visible(page, "too noisy to classify, so no beats", 5000):
            problems.append("unreadable trace not explained")
        if not visible(page, "whole recording", 3000):
            problems.append('"Need a person" does not say the whole recording')
        return status, problems
    if not visible(page, "recorded answer", 3000):
        problems.append("summary not from recorded Gemma")

    # Findings: the table, one explanation, and (once per record and scenario) the clinician override.
    page.get_by_role("tab", name="🔎 Findings").click()
    if not visible(page, "Each row is one window", 15000):
        problems.append("findings table missing")
    page.get_by_role("button", name="Explain this finding").click()
    if not visible(page, "Explained by", 30000):
        problems.append("Explain this finding gave nothing")
    elif visible(page, "Explained by ?"):
        problems.append('explanation says "Explained by ?"')
    if override:
        page.wait_for_timeout(2000)  # let the explanation's redraw finish before typing into the form
        page.get_by_label("Your name").fill("Dr Check")
        page.get_by_label("Reason").fill("automated demo check")
        page.get_by_role("button", name="Override").click()
        if not visible(page, "Overridden", 20000):
            problems.append("override not recorded")

    page.get_by_role("tab", name="⚖️ Adapter vs text").click()
    if not visible(page, "Prompt tokens Gemma read", 30000):
        problems.append("Adapter vs text did not fill in")

    # Ask: click every suggested question in the order offered; each must add a question and an answer.
    page.get_by_role("tab", name="🩺 Ask").click()
    if not visible(page, "Try one of these", 15000):
        problems.append("no suggested questions")
    asked = 0
    while asked < 10 and visible(page, "Try one of these", 3000):
        panel = page.get_by_role("tabpanel").filter(has_text="Try one of these")
        button = panel.get_by_role("button").filter(has_text=re.compile(r"\w")).first  # first offered, in order
        q = button.inner_text().strip()
        before = page.locator('[data-testid="stChatMessage"]').count()
        button.click()
        try:
            page.wait_for_function(
                f"document.querySelectorAll('[data-testid=\"stChatMessage\"]').length >= {before + 2}", timeout=T)
        except Exception:  # noqa: BLE001
            problems.append(f"no answer to {q!r}")
            break
        asked += 1
        page.get_by_role("tab", name="🩺 Ask").click()  # stay on the tab across the rerun
    problems += errors(page)
    return status, problems


def controls(page: Page, base: str, rec: str, scen: str) -> list[str]:
    """Every control on a completed run's page, clicked: conversation panel and its filter, ECG zoom, each window's
    explanation, the override and its audit entry, a free-text question; then the worklist row and the sidebar."""
    page.goto(base)
    page.get_by_text("Analyse a recording").wait_for(timeout=T)
    pick(page, "Record", rec)
    pick(page, "Mode", "balanced")
    pick(page, "Stress scenario", scen)
    page.get_by_role("button", name="Start").click()
    page.wait_for_url("**/run?run=*", timeout=T)
    page.get_by_text("Audit trail").first.wait_for(timeout=T)
    run_url, problems = page.url, []

    page.get_by_text("The agents' conversation").first.click()
    if not visible(page, "perception agent · ", 15000):
        problems.append("conversation panel does not open")
    page.get_by_text("Messages and guardrails only").first.click()
    page.wait_for_timeout(1500)
    if not visible(page, "perception agent · ", 10000):
        problems.append("conversation filter breaks the panel")
    page.get_by_text("The agents' conversation").first.click()

    slider = page.get_by_role("slider").first
    slider.focus()
    page.keyboard.press("ArrowRight")
    page.wait_for_timeout(1500)
    if not visible(page, "Markers are the perception agent's beat labels", 15000):
        problems.append("ECG zoom breaks the chart")

    page.get_by_role("tab", name="🔎 Findings").click()
    page.get_by_text("Each row is one window").wait_for(timeout=T)
    box = page.locator('[data-testid="stSelectbox"]').filter(has_text="Window").first
    box.get_by_role("combobox").click()
    windows = [o.inner_text().strip() for o in page.get_by_role("option").all()]
    page.keyboard.press("Escape")
    for w in windows:
        pick(page, "Window", w)
        page.get_by_text(f"Clinician override · {w} is").wait_for(timeout=T)  # the page has redrawn for w
        page.get_by_role("button", name="Explain this finding").click()
        if not visible(page, "Explained by", 30000) or visible(page, "Explained by ?"):
            problems.append(f"{w}: explanation missing or unattributed")
    page.wait_for_timeout(2000)  # let the last explanation's redraw finish before typing into the form
    page.get_by_label("Your name").fill("Dr Check")
    page.get_by_label("Reason").fill("automated controls check")
    page.get_by_role("button", name="Override").click()
    if not visible(page, "Overridden", 20000):
        problems.append("override not recorded")
    page.get_by_role("tab", name="🧾 Audit trail").click()
    if not visible(page, "Clinician decisions", 15000):
        problems.append("override missing from the audit trail")

    page.get_by_role("tab", name="⚖️ Adapter vs text").click()
    if not visible(page, "Prompt tokens per window", 30000):
        problems.append("Adapter vs text chart missing")

    page.get_by_role("tab", name="🩺 Ask").click()
    before = page.locator('[data-testid="stChatMessage"]').count()
    chat = page.get_by_placeholder("Ask a question about this recording")
    chat.fill("What was the median heart rate?")
    chat.press("Enter")
    try:
        page.wait_for_function(
            f"document.querySelectorAll('[data-testid=\"stChatMessage\"]').length >= {before + 2}", timeout=T)
    except Exception:  # noqa: BLE001
        problems.append("free-text question got no answer")

    # Worklist: the run is listed, and clicking its row opens it again.
    page.get_by_role("link", name="Worklist").click()
    page.get_by_text("Click a row to open the recording.").wait_for(timeout=T)
    grid = page.locator('[data-testid="stDataFrame"]').first
    bb = grid.bounding_box()
    page.mouse.click(bb["x"] + 18, bb["y"] + 35 + 17)  # first row's selection cell (header is 35 px, rows 35 px)
    try:
        page.wait_for_url("**/run?run=*", timeout=20000)
    except Exception:  # noqa: BLE001
        problems.append("clicking a worklist row does not open the run")
    page.get_by_role("link", name="How it works").click()
    if not visible(page, "How it works", 20000):
        problems.append("How it works does not render")
    problems += errors(page)
    page.goto(run_url)
    return problems


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8501/")
    ap.add_argument("--records", nargs="*", help="default: every record the site offers")
    ap.add_argument("--controls", action="store_true",
                    help="click every control on two runs per record (no scenario, noise_burst) instead")
    a = ap.parse_args()
    failures, n, t0 = [], 0, time.time()
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.goto(a.url)
        page.get_by_text("Analyse a recording").wait_for(timeout=T)
        records = a.records or options(page, "Record")
        modes, scenarios = options(page, "Mode"), options(page, "Stress scenario")
        if a.controls:
            for rec in records:
                for scen in ("none", "noise_burst"):
                    n += 1
                    try:
                        problems = controls(page, a.url, rec, scen)
                    except Exception as exc:  # noqa: BLE001
                        problems = [f"{type(exc).__name__}: {str(exc)[:160]}"]
                    print(f"{time.time() - t0:6.0f}s controls {rec} {scen:11s} "
                          + ("ok" if not problems else "FAIL: " + "; ".join(problems)), flush=True)
                    failures += [f"controls {rec} {scen}: {p_}" for p_ in problems]
            browser.close()
            print("\n".join(failures) or f"every control works on all {n} runs")
            sys.exit(1 if failures else 0)
        print(f"site offers: records {records}, modes {modes}, scenarios {scenarios}", flush=True)
        for rec in records:
            for scen in scenarios:
                for i, mode in enumerate(modes):
                    n += 1
                    try:
                        status, problems = check(page, a.url, rec, scen, mode, override=i == 0)
                    except Exception as exc:  # noqa: BLE001 - a hang or crash is a failure of this combination
                        status, problems = "error", [f"{type(exc).__name__}: {str(exc)[:160]}"]
                    print(f"{time.time() - t0:6.0f}s {rec} {scen:11s} {mode:10s} {status:10s} "
                          + ("ok" if not problems else "FAIL: " + "; ".join(problems)), flush=True)
                    failures += [f"{rec} {scen} {mode}: {p_}" for p_ in problems]
        # Pages outside a run.
        page.goto(a.url)
        if not visible(page, "Click a row to open the recording", 30000):
            failures.append("worklist does not list the runs")
        page.get_by_role("link", name="How it works").click()
        page.wait_for_timeout(3000)
        failures += [f"How it works: {e}" for e in errors(page)]
        browser.close()
    print("\n".join(failures) or f"all {n} combinations and pages work in the browser")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
