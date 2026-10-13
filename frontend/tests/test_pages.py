"""Each page renders without errors for each kind of run, using a fake API (no backend needed).

Streamlit's AppTest runs a page script headlessly and exposes what it drew. The pages import ``api``; a fake module
is put in its place, so these tests check the UI logic only.
"""
import sys
import types
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

FRONTEND = Path(__file__).resolve().parents[1]
STATUS = {"receiver": "offline", "receiver_health": {"ok": True, "mode": "offline-rule"}, "planner": "rule",
          "classifier": "rule", "uploads": False, "max_duration_s": 600.0, "disclaimer": "x"}
EVENT = {"classification": {"label": "V"}}
FINDING = {"window_id": "w000", "final_tier": "urgent", "screening_tier": "urgent", "resolution": "accepted",
           "needs_human_review": False, "n": 50, "abnormal": 4,
           "attempts": [{"channel": "adapter", "tier": "priority", "verdict": "under_triage", "source": "gemma"},
                        {"channel": "filtered", "tier": "urgent", "verdict": "agree", "source": "gemma"}]}
WINDOW = {"id": "w000", "start": 0, "n": 50, "abnormal": 4, "classes": {"N": 46, "V": 4}, "rule_tier": "urgent",
          "t_start_s": 0.5, "t_end_s": 40.0, "noise": 0.01, "readable": True}


def run_doc(status):
    complete = status == "complete"
    return {"id": "r1", "status": status, "params": {"label": "MIT-BIH 233, 0-300 s", "mode": "balanced"},
            "overall_tier": "urgent" if complete else None,
            "report": {"overall_tier": "urgent", "needs_human_review": [], "quality_warnings": ["noise level 0.04"],
                       "narrative": "Summary. Research prototype, not clinical advice.", "narrative_source": "gemma"}
            if complete else ({"status": "failed", "error": "recording unreadable: too noisy"}
                              if status == "unreadable" else None),
            "findings": {"w000": FINDING} if complete else {}, "summary": {"beats": 300, "duration_s": 300.0}
            if status != "unreadable" else {}, "windows": [WINDOW] if complete else [],
            "answers": [{"question": "Why?", "text": "Because.", "type": "explain", "citations": ["m00004"],
                         "data": {}}] if complete else [], "overrides": [],
            "comparisons": {"w000": {
                "adapter": {"tier": "priority", "source": "replay", "prompt_tokens": 713, "decision_ms": 1290.0,
                            "energy_j": 85.0},
                "filtered": {"tier": "urgent", "source": "replay", "prompt_tokens": 2480, "decision_ms": 1610.0,
                             "energy_j": 104.0}}} if complete else {}}


def fake_api(status="complete"):
    m = types.ModuleType("api")

    class ApiError(RuntimeError):
        def __init__(self, s, d):
            super().__init__(d)
            self.status, self.detail = s, d

    m.ApiError = ApiError
    m.status = lambda: STATUS
    m.records = lambda: [{"id": "233", "duration_s": 1805.6}, {"id": "100", "duration_s": 1805.6}]
    m.scenarios = lambda: ["noise_burst", "noise_10db"]
    m.worklist = lambda: [{"id": "r1", "status": status, "overall_tier": "urgent", "source": "MIT-BIH 233",
                           "needs_human_review": [], "quality_warnings": [], "error": None, "overrides": 0,
                           "created": 0.0}]
    m.run = lambda rid: run_doc(status)
    m.steps = lambda rid, after=-1: {"steps": [
        {"seq": 0, "t_ms": 1.0, "agent": "orchestrator", "kind": "message", "title": "x",
         "data": {"message": {"from": "perception", "to": "reasoning", "intent": "window",
                              "content": {"window_id": "w000", "channel": "adapter", "reason": "flat cost"}}}},
        {"seq": 1, "t_ms": 2.0, "agent": "reasoning", "kind": "guardrail", "title": "Rejected w000", "data": {}}],
        "status": status, "done": True, "total": 2}
    m.signal = lambda rid, a, b, n=3000: {"t": [0.0, 1.0, 2.0], "v": [0.0, 1.0, 0.0], "duration_s": 300.0,
                                          "beats": [{"i": 0, "t": 1.0, "label": "V", "confidence": 0.97,
                                                     "tier": "urgent", "reference": "V", "window": "w000"}]}
    m.audit = lambda rid: [{"verdict": "delivered", "from": "perception", "to": "reasoning", "intent": "window",
                            "content": {"window_id": "w000", "channel": "adapter"}, "vectors_sha256": "ab" * 32}]
    return m


@pytest.fixture
def page(monkeypatch):
    monkeypatch.syspath_prepend(str(FRONTEND))

    def make(script, status="complete"):
        monkeypatch.setitem(sys.modules, "api", fake_api(status))
        at = AppTest.from_file(str(FRONTEND / script), default_timeout=20)
        at.session_state["status"] = STATUS
        at.session_state["run_id"] = "r1"
        return at.run()
    return make


def test_worklist(page):
    at = page("views/worklist.py")
    assert not at.exception
    assert "Worklist" in at.title[0].value and len(at.dataframe) == 1


@pytest.mark.parametrize("status", ["complete", "unreadable"])
def test_recording_page(page, status):
    at = page("views/run.py", status)
    assert not at.exception, at.exception
    if status == "unreadable":
        assert any("Not triaged" in e.value for e in at.error)  # says so, never shows a tier
        assert next(m for m in at.metric if m.label == "Need a person").value == "whole recording"
    else:
        assert [m.value for m in at.metric][0].endswith("urgent")
        assert any("noise level" in w.value for w in at.warning)
        assert len(at.tabs) == 5
        tokens = next(m for m in at.metric if m.label.startswith("Prompt tokens"))
        assert tokens.value == "adapter 713" and tokens.delta.startswith("-71%")  # (713 - 2480) / 2480


def test_about(page):
    assert not page("views/about.py").exception


@pytest.mark.parametrize("receiver,auto", [("replay", True), ("http", False)])
def test_adapter_tab_fills_in_by_itself_only_with_recorded_answers(monkeypatch, receiver, auto):
    """Recorded answers: measuring is free, so the tab fills in at once. Live GPU: one Gemma call per window, so it
    waits for the button."""
    monkeypatch.syspath_prepend(str(FRONTEND))
    fake, doc, calls = fake_api("complete"), run_doc("complete"), []
    measured, doc["comparisons"] = doc["comparisons"], {}

    def compare(rid, wid):  # like the API: once measured, the run carries the comparison
        calls.append(wid)
        doc["comparisons"][wid] = measured[wid]
        return measured[wid]

    fake.compare, fake.run = compare, (lambda rid: doc)
    monkeypatch.setitem(sys.modules, "api", fake)
    at = AppTest.from_file(str(FRONTEND / "views/run.py"), default_timeout=20)
    at.session_state["status"] = {**STATUS, "receiver": receiver}
    at.session_state["run_id"] = "r1"
    at.run()
    assert not at.exception, at.exception
    assert calls == (["w000"] if auto else [])  # measured exactly once, and only where it is free
    has_button = any(b.label.startswith("Measure both channels") for b in at.button)
    assert has_button != auto
    assert any(m.label.startswith("Prompt tokens") for m in at.metric) == auto


def test_suggested_question_buttons_ask_exactly_that_question(monkeypatch):
    monkeypatch.syspath_prepend(str(FRONTEND))
    fake, doc, asked = fake_api("complete"), run_doc("complete"), []
    doc["suggestions"] = ["Why is w000 urgent?", "How does this system work?"]
    doc["answers"] = [{"question": "Why is w000 urgent?", "text": "Because.", "type": "explain", "citations": [],
                       "data": {}}]
    fake.run = lambda rid: doc
    fake.ask = lambda rid, q: asked.append(q) or {}
    monkeypatch.setitem(sys.modules, "api", fake)
    at = AppTest.from_file(str(FRONTEND / "views/run.py"), default_timeout=20)
    at.session_state["status"] = STATUS
    at.session_state["run_id"] = "r1"
    at.run()
    labels = [b.label for b in at.button]
    assert "How does this system work?" in labels and "Why is w000 urgent?" not in labels  # already asked: hidden
    next(b for b in at.button if b.label == "How does this system work?").click().run()
    assert not at.exception, at.exception
    assert asked == ["How does this system work?"]  # the recorded wording, verbatim


def test_the_window_review_suggestion_comes_last(monkeypatch):
    monkeypatch.syspath_prepend(str(FRONTEND))
    fake, doc = fake_api("complete"), run_doc("complete")
    doc["suggestions"] = ["How does this system work?", "Can you look at w003?"]
    fake.run = lambda rid: doc
    monkeypatch.setitem(sys.modules, "api", fake)
    at = AppTest.from_file(str(FRONTEND / "views/run.py"), default_timeout=20)
    at.session_state["status"], at.session_state["run_id"] = STATUS, "r1"
    at.run()
    labels = [b.label for b in at.button]
    assert "How does this system work?" in labels and "Can you look at w003?" not in labels  # not yet
    doc["answers"] = [{"question": "How does this system work?", "text": "x", "type": "system", "citations": [],
                       "data": {}}]
    at.run()
    assert "Can you look at w003?" in [b.label for b in at.button]  # now it is the only one left


def test_a_refused_start_says_why_above_the_form(monkeypatch):
    """The public demo's rate limit refuses a start; the reason must be visible, not hidden in a collapsed panel."""
    monkeypatch.syspath_prepend(str(FRONTEND))
    fake = fake_api("complete")

    def refuse(*a, **k):
        raise fake.ApiError(429, "too many run requests; try again in a minute")

    fake.create_run = refuse
    monkeypatch.setitem(sys.modules, "api", fake)
    at = AppTest.from_file(str(FRONTEND / "views/worklist.py"), default_timeout=20)
    at.session_state["status"], at.session_state["has_runs"] = STATUS, True  # runs exist: the panel starts closed
    at.run()
    at.button[0].click().run()  # Start (the form's submit button)
    assert not at.exception, at.exception
    assert any("try again in a minute" in e.value for e in at.error)
    assert at.expander[0].proto.expanded  # the form stays open for the next try


@pytest.mark.parametrize("receiver", ["replay", "http"])
def test_refused_questions_and_measurements_say_why_and_do_not_loop(monkeypatch, receiver):
    monkeypatch.syspath_prepend(str(FRONTEND))
    fake, doc, calls = fake_api("complete"), run_doc("complete"), []
    doc["comparisons"], doc["suggestions"] = {}, ["How does this system work?"]

    def refuse(*a, **k):
        calls.append(a)
        raise fake.ApiError(429, "too many requests; try again in a minute")

    fake.run, fake.compare, fake.ask = (lambda rid: doc), refuse, refuse
    monkeypatch.setitem(sys.modules, "api", fake)
    at = AppTest.from_file(str(FRONTEND / "views/run.py"), default_timeout=20)
    at.session_state["status"] = {**STATUS, "receiver": receiver}
    at.session_state["run_id"] = "r1"
    at.run()
    assert not at.exception, at.exception
    assert len(calls) == (1 if receiver == "replay" else 0)  # the automatic fill-in tries once, not in a loop
    if receiver == "replay":
        assert any("Could not measure" in e.value for e in at.error)
    next(b for b in at.button if b.label == "How does this system work?").click().run()
    assert any("Could not ask" in e.value for e in at.error)  # the refusal survives the redraw
    assert len(calls) == (2 if receiver == "replay" else 1)  # one ask; no second automatic measurement
