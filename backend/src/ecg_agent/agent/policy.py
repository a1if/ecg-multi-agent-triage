"""The orchestrator's message policy: what each agent may say, to whom, and in what shape (design §4, §5.1, §6a).

Every message is checked before delivery. A refused message is never delivered; its sender is told why, so it can
carry on (the reasoning agent records the refusal in its history, like any other failure).

Checks, in order:
  1. allowlist      (sender, recipient, intent, performative) must be a known combination;
  2. correlation    a reply (window, detail, or a failure answering a request) must answer an open request from
                    the agent it goes to; requests are closed by their reply;
  3. content        required fields per intent; a known window id; a known channel;
  4. latent envelope for adapter windows: N x input_dim, N <= max_events, one row per event, finite, within the
                    bounds the architecture guarantees (LSTM hidden state in [-1, 1]; side inputs clipped to [-3, 3]);
                    and the version contract: the vectors come from the sender the adapter was trained for.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from ecg_agent.agent.protocol import Message

CHANNELS = {"compact", "filtered", "adapter"}

ALLOWED = {
    ("perception", "reasoning", "record_ready", "inform"),
    ("perception", "reasoning", "record_ready", "failure"),
    ("perception", "reasoning", "window", "inform"),
    ("perception", "reasoning", "detail", "inform"),
    ("perception", "reasoning", "request_window", "failure"),
    ("perception", "reasoning", "request_resend", "failure"),
    ("perception", "reasoning", "request_detail", "failure"),
    ("reasoning", "perception", "request_window", "request"),
    ("reasoning", "perception", "request_resend", "request"),
    ("reasoning", "perception", "request_detail", "request"),
    ("reasoning", "orchestrator", "done", "inform"),
    ("clinician", "reasoning", "ask", "request"),  # design 6c: the clinician is a third participant
    ("reasoning", "clinician", "answer", "inform"),
}
REPLIES = {"window": {"request_window", "request_resend"}, "detail": {"request_detail"}, "answer": {"ask"}}
REQUIRED = {
    "record_ready": {"summary", "candidates"},
    "request_window": {"window_id"},
    "request_resend": {"window_id", "channel"},
    "request_detail": {"window_id"},
    "window": {"window_id", "channel", "payload", "events", "screening"},
    "detail": {"window_id", "beats"},
    "done": {"report"},
    "ask": {"question"},
    "answer": {"text", "type"},
}
# Bounds the architecture guarantees for each adapter input column: 32 LSTM hidden-state values (tanh-bounded),
# then heart rate and RR z-scores (clipped to [-3, 3]) and the run-length feature (log1p(min(run, 20)) / log 4).
CONTEXT_BOUND, SIDE_BOUND = 1.0 + 1e-4, 3.0 + 1e-4


@dataclass
class Verdict:
    ok: bool
    reason: str = ""


@dataclass
class Policy:
    manifest: dict
    window_ids: set[str] = field(default_factory=set)
    open_requests: dict[str, Message] = field(default_factory=dict)  # request id -> request

    @classmethod
    def from_manifest(cls, path: str | Path) -> Policy:
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def check(self, m: Message) -> Verdict:
        if (m.sender, m.recipient, m.intent, m.performative) not in ALLOWED:
            return Verdict(False, f"{m.sender} may not send {m.performative} {m.intent} to {m.recipient}")
        if m.performative == "failure" and m.intent != "record_ready":
            return self._reply(m, {m.intent})
        if m.performative != "failure":
            missing = REQUIRED.get(m.intent, set()) - set(m.content)
            if missing:
                return Verdict(False, f"{m.intent} is missing {sorted(missing)}")
        if m.intent in REPLIES:
            v = self._reply(m, REPLIES[m.intent])
            if not v.ok:
                return v
        c = m.content
        if m.intent == "record_ready" and m.performative == "inform":
            self.window_ids = {w["id"] for w in c["candidates"]}
        if c.get("window_id") is not None and m.intent != "record_ready" and c["window_id"] not in self.window_ids:
            return Verdict(False, f"unknown window {c['window_id']!r}")
        if "channel" in c and c["channel"] not in CHANNELS:
            return Verdict(False, f"unknown channel {c['channel']!r}")
        if m.intent == "window" and c["channel"] == "adapter":
            return self._envelope(c)
        return Verdict(True)

    def accepted(self, m: Message) -> None:
        """Book-keeping once a message is delivered: open a request, or close the one it answers."""
        if m.performative == "request":
            self.open_requests[m.id] = m
        elif m.in_reply_to:
            self.open_requests.pop(m.in_reply_to, None)

    def _reply(self, m: Message, answers: set[str]) -> Verdict:
        req = self.open_requests.get(m.in_reply_to or "")
        if req is None:
            return Verdict(False, f"{m.intent} does not answer an open request")
        if req.intent not in answers or req.sender != m.recipient:
            return Verdict(False, f"{m.intent} cannot answer {req.intent} from {req.sender}")
        asked, said = req.content.get("window_id"), m.content.get("window_id")
        if asked is not None and said is not None and asked != said:
            return Verdict(False, "reply is about a different window than its request")
        return Verdict(True)

    def _envelope(self, c: dict) -> Verdict:
        a = self.manifest["adapter"]
        vec = c["payload"].get("vectors")
        if not isinstance(vec, list) or not vec:
            return Verdict(False, "adapter window without vectors")
        n, dim = len(vec), a["input_dim"]
        if n > a["max_events"]:
            return Verdict(False, f"{n} events exceed the adapter's {a['max_events']}")
        if n != len(c["events"]) or n != c["screening"].get("n", n):
            return Verdict(False, f"{n} vector rows for {len(c['events'])} events")
        if any(len(row) != dim for row in vec):
            return Verdict(False, f"vectors must be {n} x {dim}")
        ctx = self.manifest["sender"]["context_dim"]
        for row in vec:
            for j, x in enumerate(row):
                if not math.isfinite(x):
                    return Verdict(False, "non-finite value in the vectors")
                if abs(x) > (CONTEXT_BOUND if j < ctx else SIDE_BOUND):
                    return Verdict(False, f"value {x:.3g} in column {j} is outside the range the sender can produce")
        sha = c["payload"].get("sender_sha256")
        if sha != a["trained_for_sender_sha256"]:
            return Verdict(False, "version contract: vectors are not from the sender this adapter was trained for")
        return Verdict(True)


def vectors_sha256(vec: list[list[float]]) -> str:
    """Fingerprint of a latent message for the audit log (the exact message can be re-created and re-run)."""
    return hashlib.sha256(json.dumps(vec, separators=(",", ":")).encode()).hexdigest()
