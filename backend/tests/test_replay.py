"""Recorded answers: an answer is served only for exactly what the receiver would have seen, adapter included."""
import asyncio

from ecg_agent.receiver.base import TriageRequest, TriageResult
from ecg_agent.receiver.clients import ReplayReceiver

OLD, NEW = "a" * 64, "b" * 64


def adapter_request() -> TriageRequest:
    events = [{"clinical_flags": {"consecutive_abnormal_beats": 0}}]
    return TriageRequest(channel="adapter", events=events, vectors=[[0.1] * 35], mode="decision")


class Live:
    """A stand-in GPU service that records what it was asked."""

    name = "live"

    def __init__(self):
        self.calls = 0

    async def triage(self, req):
        self.calls += 1
        return TriageResult(channel=req.channel, tier="urgent", source="gemma", latency_ms=1.0, mode=req.mode)


def test_adapter_answers_are_keyed_by_the_adapter():
    req = adapter_request()
    assert req.cache_key(OLD) != req.cache_key(NEW)  # same vectors, different adapter: different virtual tokens
    assert req.cache_key(OLD) == req.cache_key(OLD)


def test_a_promoted_adapter_never_gets_the_old_adapters_answers(tmp_path):
    store = tmp_path / "answers.jsonl"
    old_live = Live()
    asyncio.run(ReplayReceiver(store, live=old_live, adapter_sha256=OLD).triage(adapter_request()))
    assert old_live.calls == 1 and store.exists()

    replay_old = ReplayReceiver(store, live=None, adapter_sha256=OLD)  # same adapter: served from the recording
    assert asyncio.run(replay_old.triage(adapter_request())).source == "replay"

    new_live = Live()  # after a promotion the recording is a miss, so the new adapter answers
    asyncio.run(ReplayReceiver(store, live=new_live, adapter_sha256=NEW).triage(adapter_request()))
    assert new_live.calls == 1
