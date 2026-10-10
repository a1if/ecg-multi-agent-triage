"""Record the README demo GIF: one run from start to audit trail, with a caption on every frame.

    python frontend/scripts/record_demo_gif.py [--url http://127.0.0.1:8501/] [--out docs/images/demo.gif]

Drives the app with Playwright (installed Chrome, like capture_screenshots.py), takes a viewport screenshot at each
step of the story, and assembles them with Pillow, so no ffmpeg is needed. Run it against a fresh app (the local
demo configuration or the live demo) so the worklist starts empty.
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path

from capture_screenshots import TIMEOUT, pick
from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import Page, sync_playwright

ROOT = Path(__file__).resolve().parents[2]
VIEW = {"width": 1280, "height": 760}
WIDTH = 960  # GIF width; READMEs render at most ~900 px, and smaller frames keep the file a few MB
BANNER = 44


def font(size: int):
    for name in ("segoeuib.ttf", "DejaVuSans-Bold.ttf", "Arial Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


class Reel:
    """Frames with captions and durations; saved as one looping GIF."""

    def __init__(self, page: Page):
        self.page, self.frames, self.font = page, [], font(20)

    def snap(self, caption: str, ms: int, settle: int = 800) -> None:
        self.page.wait_for_timeout(settle)
        img = Image.open(io.BytesIO(self.page.screenshot())).convert("RGB")
        img = img.resize((WIDTH, round(img.height * WIDTH / img.width)), Image.LANCZOS)
        out = Image.new("RGB", (WIDTH, img.height + BANNER), "#0f172a")
        out.paste(img, (0, BANNER))
        ImageDraw.Draw(out).text((16, BANNER // 2), caption, fill="#f8fafc", font=self.font, anchor="lm")
        self.frames.append((out, ms))
        print(f"frame {len(self.frames):2d} ({ms} ms): {caption}")

    def save(self, path: Path) -> None:
        # One shared adaptive palette keeps colours stable between frames (no flicker) and the file small.
        frames = [f.quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
                  for f, _ in self.frames]
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=[ms for _, ms in self.frames],
                       loop=0, optimize=True, disposal=1)
        print(f"saved {path} ({path.stat().st_size / 1e6:.1f} MB, {len(frames)} frames)")


def scroll_to(page: Page, text: str) -> None:
    """Put the element near the top of the view (Streamlit scrolls an inner container, so use scrollIntoView)."""
    page.get_by_text(text).first.evaluate("e => e.scrollIntoView({block: 'start'})")
    page.mouse.move(VIEW["width"] // 2, VIEW["height"] // 2)
    page.mouse.wheel(0, -60)  # keep a little context above the target


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8501/")
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "images" / "demo.gif")
    a = ap.parse_args()
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport=VIEW, color_scheme="light")
        reel = Reel(page)

        # 1. The clinician picks a recording.
        page.goto(a.url)
        page.get_by_text("Analyse a recording").wait_for(timeout=TIMEOUT)
        pick(page, "Record", "233")
        reel.snap("1  A clinician picks a recording (MIT-BIH record 233, 5 minutes)", 2200)

        # 2. The two agents work; the page follows the run live.
        page.get_by_role("button", name="Start").click()
        page.wait_for_url("**/run?run=*", timeout=TIMEOUT)
        done = page.get_by_text("Audit trail").first
        for _ in range(20):
            page.wait_for_timeout(500)
            if done.is_visible():
                break
            reel.snap("2  Perception agent screens the signal; reasoning agent plans and triages", 1800, settle=0)
            if done.is_visible():  # finished while the frame was taken: it may already show the result
                reel.frames.pop()
                break
        done.wait_for(timeout=TIMEOUT)
        page.get_by_text("Most urgent tier").wait_for(timeout=TIMEOUT)
        reel.snap("3  Result: most urgent tier, windows reviewed, quality warnings", 2600)

        # 3. The conversation between the agents, through the orchestrator.
        page.get_by_text("The agents' conversation").click()
        page.get_by_text("perception agent · ").first.wait_for(timeout=TIMEOUT)
        scroll_to(page, "The agents' conversation")
        reel.snap("4  Every message between the agents went through the orchestrator's checks", 3000)
        page.get_by_text("The agents' conversation").click()

        # 4. Questions: one answered with a citation, one refused before any LLM sees it.
        page.get_by_role("tab", name="🩺 Ask").click()
        chat = page.get_by_placeholder("Ask a question about this recording")
        for q, caption in (("Why is w000 urgent?", "5  Ask why: answered by Gemma, citing the agents' messages"),
                           ("Should she take amiodarone?", "6  Medical advice is refused before any LLM sees it")):
            chat.fill(q)
            chat.press("Enter")
            page.get_by_text(q).last.wait_for(timeout=TIMEOUT)
            page.wait_for_timeout(3000)
            chat.scroll_into_view_if_needed()
            reel.snap(caption, 3000)

        # 5. The clinician overrides a tier: a recorded human decision, never something the agents can do.
        page.get_by_role("tab", name="🔎 Findings").click()
        page.get_by_text("Each row is one window").wait_for(timeout=TIMEOUT)
        scroll_to(page, "Clinician override")
        page.get_by_label("Your name").fill("Dr Demo")
        page.get_by_label("Reason").fill("artefact on review of the strip")
        reel.snap("7  The clinician can change a tier, with their name and a reason", 2400)
        page.get_by_role("button", name="Override").click()
        page.get_by_text("recorded in the audit trail").wait_for(timeout=TIMEOUT)
        scroll_to(page, "Clinician override")
        reel.snap("7  The clinician can change a tier, with their name and a reason", 2000, settle=300)

        # 6. The audit trail ends with that decision.
        page.get_by_role("tab", name="🧾 Audit trail").click()
        page.get_by_text("Clinician decisions").wait_for(timeout=TIMEOUT)
        scroll_to(page, "Clinician decisions")
        reel.snap("8  The audit trail records every check and every human decision", 3500)

        browser.close()
    reel.save(a.out)


if __name__ == "__main__":
    main()
