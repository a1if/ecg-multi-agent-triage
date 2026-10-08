"""The orchestrator's message policy: each test tries to make an agent step outside its job."""
import math

import pytest

from ecg_agent.agent.protocol import Message

SENDER_SHA = "e0e953787691c522b87c0eb03829f76ab8919f04274835f77bd772cef3d8861d"


def event():
    return {"classification": {"label": "N", "confidence": 0.99}, "clinical_flags": {"consecutive_abnormal_beats": 0},
            "signal_features": {"heart_rate_bpm": 75.0, "rr_interval_ms": 800.0}}


def msg(sender, recipient, perf, intent, content=None, reply=None):
    return Message(sender=sender, recipient=recipient, performative=perf, intent=intent, content=content or {},
                   in_reply_to=reply)


def ready(policy):
    m = msg("perception", "reasoning", "inform", "record_ready", {"summary": {}, "candidates": [{"id": "w000"}]})
    assert policy.check(m).ok
    policy.accepted(m)


def request(policy, intent="request_window", **content):
    m = msg("reasoning", "perception", "request", intent, {"window_id": "w000", **content})
    assert policy.check(m).ok
    policy.accepted(m)
    return m


def window(req, n=10, channel="adapter", vectors=None, sha=SENDER_SHA):
    payload = {"channel": channel}
    if channel == "adapter":
        payload["vectors"] = vectors if vectors is not None else [[0.1] * 32 + [0.0, 0.0, 0.5] for _ in range(n)]
        payload["sender_sha256"] = sha
    return msg("perception", "reasoning", "inform", "window",
               {"window_id": "w000", "channel": channel, "payload": payload, "events": [event()] * n,
                "screening": {"rule_tier": "routine", "abnormal": 0, "n": n}}, req.id)


def test_normal_exchange_is_delivered(policy):
    ready(policy)
    req = request(policy)
    assert policy.check(window(req)).ok


@pytest.mark.parametrize("sender,recipient,perf,intent", [
    ("perception", "reasoning", "request", "request_resend"),  # perception may not make requests
    ("reasoning", "perception", "inform", "window"),  # reasoning may not send data
    ("reasoning", "reasoning", "request", "request_window"),  # nobody talks to themselves
    ("perception", "orchestrator", "inform", "done"),  # only reasoning ends the conversation
])
def test_allowlist_refuses_messages_outside_an_agents_job(policy, sender, recipient, perf, intent):
    v = policy.check(msg(sender, recipient, perf, intent, {"window_id": "w000", "report": {}, "channel": "filtered"}))
    assert not v.ok and "may not send" in v.reason


def test_unsolicited_window_is_refused(policy):
    ready(policy)
    fake = msg("reasoning", "perception", "request", "request_window", {"window_id": "w000"})  # never delivered
    v = policy.check(window(fake))
    assert not v.ok and "open request" in v.reason


def test_a_request_is_answered_once(policy):
    ready(policy)
    req = request(policy)
    first = window(req)
    policy.accepted(first)
    assert not policy.check(window(req)).ok


def test_unknown_window_and_channel_are_refused(policy):
    ready(policy)
    assert "unknown window" in policy.check(
        msg("reasoning", "perception", "request", "request_window", {"window_id": "w999"})).reason
    assert "unknown channel" in policy.check(
        msg("reasoning", "perception", "request", "request_resend", {"window_id": "w000", "channel": "telepathy"})).reason


@pytest.mark.parametrize("vectors,why", [
    ([[0.1] * 35 for _ in range(9)], "vector rows"),  # 9 rows for 10 events
    ([[0.1] * 34 for _ in range(10)], "x 35"),  # wrong width
    ([[math.nan] + [0.1] * 34 for _ in range(10)], "non-finite"),
    ([[1.7] + [0.1] * 34 for _ in range(10)], "outside the range"),  # LSTM hidden state cannot exceed 1
    ([[0.1] * 32 + [4.0, 0.0, 0.5] for _ in range(10)], "outside the range"),  # side input clipped to 3
])
def test_latent_envelope(policy, vectors, why):
    ready(policy)
    req = request(policy)
    v = policy.check(window(req, vectors=vectors))
    assert not v.ok and why in v.reason


def test_too_many_events_for_the_adapter(policy):
    ready(policy)
    req = request(policy)
    v = policy.check(window(req, n=51))
    assert not v.ok and "exceed" in v.reason


def test_version_contract(policy):
    ready(policy)
    req = request(policy)
    v = policy.check(window(req, sha="0" * 64))
    assert not v.ok and "version contract" in v.reason


def test_text_windows_need_no_vectors(policy):
    ready(policy)
    req = request(policy, channel="filtered")
    assert policy.check(window(req, channel="filtered")).ok
