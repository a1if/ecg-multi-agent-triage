"""Both agents end to end on CPU: offline (rule) receiver, injected receiver faults, real sender and records."""
import asyncio

import pytest
from conftest import RECORDS

from ecg_agent.agent.harness import Budget, Tracer
from ecg_agent.agent.orchestrator import Orchestrator
from ecg_agent.agent.perception_agent import PerceptionAgent, load_mitdb
from ecg_agent.agent.planner import Decision, RulePlanner
from ecg_agent.agent.reasoning_agent import ReasoningAgent
from ecg_agent.receiver.clients import FaultyReceiver, OfflineReceiver


def build(sender, policy, receiver, record="233", seconds=120, mode="balanced", planner=None, reviews=3):
    tr, b = Tracer("test"), Budget(max_reviews=reviews)
    p = PerceptionAgent(sender, load_mitdb(RECORDS, record, 0, seconds), tr, mode, b)
    r = ReasoningAgent(receiver, planner or RulePlanner(), tr, mode, b)
    return Orchestrator(p, r, tr, policy, b), r, tr


def run(coro):
    return asyncio.run(coro)


def test_conversation_completes_with_no_refusals(sender, policy):
    orch, r, _ = build(sender, policy, OfflineReceiver())
    report = run(orch.run())
    assert orch.status == "complete" and report["overall_tier"] == "urgent"
    assert all(a["verdict"] == "delivered" for a in orch.audit)
    assert len(r.findings) == 3


def test_guardrail_catches_injected_under_triage(sender, policy):
    faulty = FaultyReceiver(OfflineReceiver(), p_under=1.0, p_unparsed=0.0, seed=0)  # every answer one tier low
    orch, r, tr = build(sender, policy, faulty, mode="throughput")
    run(orch.run())
    assert any(s["agent"] == "reasoning" and s["kind"] == "guardrail" and "Rejected" in s["title"] for s in tr.steps)
    # nothing is ever reported below the rule's tier
    for f in r.findings.values():
        assert f["final_tier"] == f["screening_tier"]
        assert f["resolution"] == "guardrail_override" or f["attempts"][-1]["verdict"] == "agree"


def test_audit_fingerprints_latent_messages(sender, policy):
    orch, _, _ = build(sender, policy, OfflineReceiver(), mode="throughput")
    run(orch.run())
    adapter = [a for a in orch.audit if a["intent"] == "window" and a["content"]["channel"] == "adapter"]
    assert adapter and all(len(a["vectors_sha256"]) == 64 for a in adapter)


def test_coverage_guard_overrules_an_early_finish(sender, policy):
    class Lazy:
        name = "lazy"

        async def decide(self, view):
            return Decision("finish", thought="nothing to see")

    orch, r, tr = build(sender, policy, OfflineReceiver(), planner=Lazy())
    run(orch.run())
    assert any("cannot finish" in s["title"] for s in tr.steps if s["kind"] == "guardrail")
    assert r.findings  # urgent windows were reviewed anyway


def test_refused_request_is_reported_back_and_run_still_completes(sender, policy):
    class Confused:
        """Asks for a window that does not exist once, then behaves."""
        name = "confused"

        def __init__(self):
            self.n = 0

        async def decide(self, view):
            self.n += 1
            if self.n == 1:
                return Decision("request_detail", {"window_id": "w999"})
            return await RulePlanner().decide(view)

    orch, r, tr = build(sender, policy, OfflineReceiver(), planner=Confused())
    run(orch.run())
    assert orch.status == "complete"
    assert any("unknown window" in h for h in r.history)  # the harness rejected it before it was even sent


def test_contract_violation_is_refused_and_text_channel_takes_over(sender, policy, monkeypatch):
    """A sender update the adapter was not trained for: the orchestrator refuses every latent message, the reasoning
    agent asks for the window over text instead, and the run still ends with every finding at or above the rule."""
    orch, r, tr = build(sender, policy, OfflineReceiver(), mode="throughput")
    monkeypatch.setattr(orch.p.sender, "checkpoint_sha256", "0" * 64)
    run(orch.run())
    refused = [a for a in orch.audit if a["verdict"] == "refused"]
    assert len(refused) == 1 and "version contract" in refused[0]["reason"]  # circuit breaker: refused only once
    assert any(s["title"] == "Stop using adapter for this run" for s in tr.steps)
    assert orch.status == "complete" and len(r.findings) == 3
    first, *rest = sorted(r.findings.values(), key=lambda f: len(f["attempts"]), reverse=True)
    assert first["attempts"][0]["verdict"] == "refused"
    for f in r.findings.values():
        assert f["attempts"][-1]["channel"] == "filtered" and f["final_tier"] == f["screening_tier"]
    assert all(len(f["attempts"]) == 1 for f in rest)  # later windows went straight to text


def test_explanations_take_the_readable_channel(sender, policy):
    orch, r, tr = build(sender, policy, OfflineReceiver(), mode="throughput")
    run(orch.run())
    wid = next(iter(r.findings))
    exp = run(orch.explain(wid))
    assert exp["justification"]
    req = [a for a in orch.audit if a["intent"] == "request_window" and a["content"].get("purpose") == "explain"]
    assert req and req[0]["content"]["channel"] == "filtered"


def test_unreadable_recording_is_never_routine(sender, policy):
    import numpy as np

    from ecg_agent.agent.perception_agent import Source

    flat = Source("upload", {"lead": np.zeros(360 * 60)}, 360.0, label="flat line")
    tr, b = Tracer("t"), Budget()
    orch = Orchestrator(PerceptionAgent(sender, flat, tr, budget=b),
                        ReasoningAgent(OfflineReceiver(), RulePlanner(), tr), tr, policy, b)
    report = run(orch.run())
    assert orch.status == "unreadable" and report.get("overall_tier") is None


@pytest.mark.parametrize("text,bad", [
    ("Beat 12 of 20 is a V beat with confidence 0.96, above 0.85.", []),
    ("Second beat of 20:100: the second of 32:32:2032", ["100", "32", "32", "2032"]),
])
def test_grounding(text, bad):
    from ecg_agent.agent.grounding import check

    ev = [{"classification": {"label": "V", "confidence": 0.957}, "clinical_flags": {"consecutive_abnormal_beats": 1},
           "signal_features": {"heart_rate_bpm": 72.0, "rr_interval_ms": 833.33}}] * 20
    assert check(text, ev) == bad
