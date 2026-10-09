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
                         "data": {}}] if complete else [], "overrides": []}


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
    else:
        assert [m.value for m in at.metric][0].endswith("urgent")
        assert any("noise level" in w.value for w in at.warning)
        assert len(at.tabs) == 4


def test_about(page):
    assert not page("views/about.py").exception
