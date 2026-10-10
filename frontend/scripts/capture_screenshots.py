"""Capture the README screenshots from a running app (the live demo by default). Regenerate after UI changes.

    python frontend/scripts/capture_screenshots.py [--url https://ecg-multi-agent-triage.streamlit.app/~/+/]

Needs `pip install playwright`; it drives the installed Chrome (channel="chrome"), so no browser download.
Streamlit draws over a websocket after the page loads, so every step waits for text to appear before capturing.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

OUT = Path(__file__).resolve().parents[2] / "docs" / "images"
TIMEOUT = 120_000  # a sleeping Community Cloud app takes a while to wake


def pick(page: Page, label: str, option: str) -> None:
    """Choose an option in the Streamlit selectbox whose label starts with ``label``."""
    box = page.locator('[data-testid="stSelectbox"]').filter(has_text=label).first
    box.get_by_role("combobox").click()  # accessibility roles are stabler than Streamlit's internal markup
    page.get_by_role("option", name=option, exact=True).click()


def start_run(page: Page, base: str, record: str, scenario: str = "none", mode: str = "balanced") -> None:
    page.goto(base)
    page.get_by_text("Analyse a recording").wait_for(timeout=TIMEOUT)
    pick(page, "Record", record)
    pick(page, "Mode", mode)
    pick(page, "Stress scenario", scenario)
    page.get_by_role("button", name="Start").click()
    page.wait_for_url("**/run?run=*", timeout=TIMEOUT)
    page.get_by_text("Most urgent tier").wait_for(timeout=TIMEOUT)
    page.get_by_text("Audit trail").first.wait_for(timeout=TIMEOUT)  # tabs appear once the run has finished


def shot(page: Page, name: str, locator=None) -> None:
    page.wait_for_timeout(1500)  # let charts and fonts settle
    path = OUT / f"{name}.png"
    (locator or page).screenshot(path=str(path))
    print("saved", path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="https://ecg-multi-agent-triage.streamlit.app/~/+/")
    base = ap.parse_args().url
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000}, color_scheme="light",
                                device_scale_factor=1.5)

        # 1. A noisy recording the perception agent refuses to triage, and a partly noisy one.
        start_run(page, base, "100", "noise_10db")
        start_run(page, base, "105", "noise_burst")

        # 2. The main example: record 233, every view.
        start_run(page, base, "233")
        shot(page, "recording")

        page.get_by_text("The agents' conversation").click()
        page.get_by_text("perception agent · ").wait_for(timeout=TIMEOUT)
        shot(page, "conversation", page.locator('[data-testid="stExpander"]').first)
        page.get_by_text("The agents' conversation").click()  # collapse again

        page.get_by_role("tab", name="🩺 Ask").click()
        chat = page.get_by_placeholder("Ask a question about this recording")
        for q in ("Why is w000 urgent?", "Should she take amiodarone?"):
            chat.fill(q)
            chat.press("Enter")
            page.get_by_text(q).last.wait_for(timeout=TIMEOUT)
            page.wait_for_timeout(4000)
        shot(page, "questions", page.get_by_role("tabpanel").filter(has_text="Ask about this recording"))

        page.get_by_role("tab", name="🔎 Findings").click()
        page.get_by_text("Each row is one window").wait_for(timeout=TIMEOUT)
        shot(page, "findings", page.get_by_role("tabpanel").filter(has_text="Each row is one window"))

        # 3. The worklist with all three runs, most in need of attention first.
        page.goto(base)
        page.get_by_text("Click a row to open the recording.").wait_for(timeout=TIMEOUT)
        page.get_by_text("Analyse a recording").click()  # collapse the form so the list is visible
        shot(page, "worklist")
        browser.close()


if __name__ == "__main__":
    main()
