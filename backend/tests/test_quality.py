"""Stopping on poor recordings and grounding the summary: each signal fault has an expected outcome."""
import asyncio

import pytest
import wfdb
from conftest import RECORDS

from ecg_agent.agent.grounding import check_narrative
from ecg_agent.agent.harness import Budget, Tracer
from ecg_agent.agent.orchestrator import Orchestrator
from ecg_agent.agent.perception_agent import PerceptionAgent, Source
from ecg_agent.agent.planner import RulePlanner
from ecg_agent.agent.reasoning_agent import ReasoningAgent
from ecg_agent.receiver.clients import OfflineReceiver
from ecg_agent.signal.stress import SCENARIOS


def lead(record="233", seconds=300):
    r = wfdb.rdrecord(str(RECORDS / record))
    return r.p_signal[:seconds * 360, r.sig_name.index("MLII")]


def intake(sender, x):
    p = PerceptionAgent(sender, Source("upload", {"MLII": x}, 360.0), Tracer("q"))
    msg = asyncio.run(p.turn([]))[0]
    return p, msg


@pytest.mark.parametrize("scenario,outcome", [
    ("clean", "ok"),
    ("inverted", "ok"),  # the classifier copes with polarity
    ("wander", "ok"),  # removed by the baseline filter
    ("noise_burst", "windows"),  # local noise: only those windows are excluded
    ("dropout_50", "warning"),  # a third flat: assessed parts go on, flat parts named
    ("noise_10db", "unreadable"),  # where the classifier starts missing ventricular beats
    ("noise_0db", "unreadable"),
    ("clipped", "unreadable"),
    ("dropout_90", "unreadable"),
])
def test_signal_faults(sender, scenario, outcome):
    x = lead()
    p, msg = intake(sender, x if scenario == "clean" else SCENARIOS[scenario](x, 360.0))
    if outcome == "unreadable":
        assert p.unreadable and msg.performative == "failure" and msg.content["error"].startswith("recording unreadable")
        return
    assert not p.unreadable and msg.intent == "record_ready" and msg.performative == "inform"
    warnings, bad = p.summary["quality_warnings"], p.summary["unreadable_windows"]
    if outcome == "ok":
        assert not warnings and not bad
    elif outcome == "windows":
        assert bad and all(not w["readable"] for w in msg.content["candidates"] if w["id"] in bad)
        assert all(w["readable"] for w in msg.content["candidates"][:len(msg.content["candidates"]) - len(bad)])
    else:
        assert any("flat" in w for w in warnings)


def run(sender, policy, x):
    tr, b = Tracer("q"), Budget(max_reviews=20)
    orch = Orchestrator(PerceptionAgent(sender, Source("upload", {"MLII": x}, 360.0), tr, budget=b),
                        ReasoningAgent(OfflineReceiver(), RulePlanner(), tr, budget=b), tr, policy, b)
    return orch, asyncio.run(orch.run())


def test_unreadable_recording_ends_the_run_without_a_tier(sender, policy):
    orch, report = run(sender, policy, SCENARIOS["noise_10db"](lead(), 360.0))
    assert orch.status == "unreadable" and "overall_tier" not in report and "too noisy" in report["error"]


def test_noisy_windows_are_never_triaged_and_are_listed_for_review(sender, policy):
    orch, report = run(sender, policy, SCENARIOS["noise_burst"](lead(), 360.0))
    bad = report["unreadable_windows"]
    assert orch.status == "complete" and bad
    assert set(bad) <= set(report["needs_human_review"])
    sent = {a["content"]["window_id"] for a in orch.audit if a["intent"] == "window"}
    assert not sent & set(bad)  # not one unreadable window reached the receiver
    assert any("too noisy" in w for w in report["quality_warnings"])


FACTS = {"overall_tier": "urgent", "beats": 417, "duration_s": 300.0, "median_heart_rate": 83.7,
         "windows_reviewed": [{"window": "w000", "tier": "urgent"}, {"window": "w002", "tier": "urgent"}],
         "windows_needing_human_review": [], "unreadable_windows": [], "quality_warnings": []}
REAL = ("The overall triage tier is urgent. The ECG contained 417 beats over a 300-second duration with a median heart "
        "rate of 83.7. All reviewed windows were classified as urgent and accepted. There were no windows needing "
        "human review. Research prototype, not clinical advice.")  # Gemma's own summary from a real run


@pytest.mark.parametrize("text,facts,problem", [
    (REAL, FACTS, None),
    (REAL.replace("83.7", "92"), FACTS, "numbers not in the facts"),
    (REAL + " Window w002 was routine.", FACTS, "tiers not in the facts"),
    (REAL, dict(FACTS, windows_needing_human_review=["w004"]), "says no window needs review"),
    (REAL.replace(" Research prototype, not clinical advice.", ""), FACTS, "missing disclaimer"),
])
def test_narrative_grounding(text, facts, problem):
    found = check_narrative(text, facts)
    assert (found == []) if problem is None else any(problem in p for p in found)
