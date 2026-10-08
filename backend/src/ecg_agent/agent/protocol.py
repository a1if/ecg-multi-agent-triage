"""Typed messages between the two agents.

The perception agent (sender) and the reasoning agent (receiver) share no memory: everything the reasoning agent
knows about the recording arrives as a message. Performatives follow the usual agent-communication split
(inform / request / failure); ``intent`` says what the message is about.

Each ``window`` message carries two parts with different readers:
  * ``payload``   what the receiver LLM sees: a text prompt (compact / filtered) or sender vectors (adapter);
  * ``screening`` the sender's own triage flags, read only by the reasoning agent's guardrail, never by the LLM.
    In the paper these flags are the reference answer and are withheld from the receiver for the same reason.
``events`` travel with the message so the inference service can render the channel's view of them; it renders only
that view (compact or filtered payload, or the adapter's side inputs), never the flags.
"""
from __future__ import annotations

import itertools
import time
from typing import Any, Literal

from pydantic import BaseModel, Field

AgentName = Literal["perception", "reasoning", "orchestrator", "clinician"]
Performative = Literal["inform", "request", "failure"]
Intent = Literal["record_ready", "request_window", "window", "request_detail", "detail", "request_resend", "done",
                 "ask", "answer"]

_ids = itertools.count()


class Message(BaseModel):
    id: str = Field(default_factory=lambda: f"m{next(_ids):05d}")
    sender: AgentName
    recipient: AgentName
    performative: Performative
    intent: Intent
    content: dict[str, Any] = Field(default_factory=dict)
    in_reply_to: str | None = None
    ts: float = Field(default_factory=time.time)

    def brief(self) -> dict:
        """What the UI timeline shows: the message without large arrays."""
        c = dict(self.content)
        if "vectors" in c.get("payload", {}):
            vec = c["payload"]["vectors"]
            c["payload"] = {**c["payload"], "vectors": f"<{len(vec)} x {len(vec[0]) if vec else 0} floats>"}
        if "events" in c:
            c["events"] = f"<{len(c['events'])} events>"
        return {"id": self.id, "from": self.sender, "to": self.recipient, "performative": self.performative,
                "intent": self.intent, "in_reply_to": self.in_reply_to, "content": c}
