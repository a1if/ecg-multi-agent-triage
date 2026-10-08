"""In-process GPU receiver: the agents talk to a ``GemmaEngine`` in the same process, through the same async
interface as ``HttpReceiver``. For local development and GPU tests; the deployed system uses the inference service."""
from __future__ import annotations

import asyncio

from ecg_agent.receiver.base import ExplainRequest, GenerateRequest, GenerateResult, TriageRequest, TriageResult


class LocalGemmaReceiver:
    name = "gemma-local"

    def __init__(self, engine):
        self.engine = engine

    async def triage(self, req: TriageRequest) -> TriageResult:
        return await asyncio.to_thread(self.engine.triage, req)

    async def explain(self, req: ExplainRequest) -> TriageResult:
        return await asyncio.to_thread(self.engine.explain, req, req.tier)

    async def generate(self, req: GenerateRequest) -> GenerateResult | None:
        return await asyncio.to_thread(self.engine.generate, req)

    async def health(self) -> dict:
        return {"ok": True, "mode": self.name}
