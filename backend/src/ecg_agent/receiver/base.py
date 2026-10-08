"""Receiver interface: anything that can answer "most urgent tier in this window" over a channel, and generate text.

Implementations: ``HttpReceiver`` (the GPU inference service), ``ReplayReceiver`` (recorded Gemma answers, so the
public demo works while the GPU is scaled to zero), ``OfflineReceiver`` (the rule itself, clearly labelled), and
``FaultyReceiver`` (injects under-triage for the agent evaluation).
"""
from __future__ import annotations

import hashlib
import json
from typing import Literal, Protocol

from pydantic import BaseModel, Field

Channel = Literal["compact", "filtered", "adapter"]


class TriageRequest(BaseModel):
    channel: Channel
    events: list[dict]
    vectors: list[list[float]] | None = None  # sender vectors, required for the adapter channel
    mode: Literal["decision", "full"] = "full"  # "decision" stops at the tier (decide early, explain later)
    sender_sha256: str | None = None  # adapter channel: which sender made the vectors (the version contract)

    def cache_key(self) -> str:
        """Content hash of what the receiver actually sees on this channel."""
        from ecg_agent.core.prompts import compact_prompt, filtered_prompt

        if self.channel == "adapter":
            body = json.dumps([[round(v, 5) for v in row] for row in self.vectors or []]) + json.dumps(
                [e["clinical_flags"]["consecutive_abnormal_beats"] for e in self.events])
        else:
            body = (compact_prompt if self.channel == "compact" else filtered_prompt)(self.events)
        return hashlib.sha256(f"{self.channel}|{self.mode}|{body}".encode()).hexdigest()


class ExplainRequest(TriageRequest):
    """The justification for a tier the receiver already decided on this window and channel."""
    tier: Literal["routine", "priority", "urgent"]


class TriageResult(BaseModel):
    channel: Channel
    tier: Literal["routine", "priority", "urgent"] | None
    justification: str = ""
    guideline_fact: str = ""
    parsed: bool = True
    prompt_tokens: int | None = None
    generated_tokens: int | None = None
    latency_ms: float | None = None
    ttft_ms: float | None = None
    decision_ms: float | None = None  # time until the tier token exists
    mode: Literal["decision", "full", "explain"] = "full"
    source: Literal["gemma", "replay", "offline-rule", "mock"] = "gemma"
    raw: str = ""


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = Field(256, le=1024)
    system: str | None = None


class GenerateResult(BaseModel):
    text: str
    prompt_tokens: int | None = None
    generated_tokens: int | None = None
    latency_ms: float | None = None
    source: Literal["gemma", "replay", "offline-rule", "mock"] = "gemma"


class Receiver(Protocol):
    name: str

    async def triage(self, req: TriageRequest) -> TriageResult: ...

    async def explain(self, req: ExplainRequest) -> TriageResult: ...

    async def generate(self, req: GenerateRequest) -> GenerateResult | None: ...

    async def health(self) -> dict: ...
