from __future__ import annotations

import asyncio
import hashlib
import json
import random
import threading
from pathlib import Path

import httpx

from ecg_agent.core.rule import RANK, TIERS, rule_answer
from ecg_agent.receiver.base import ExplainRequest, GenerateRequest, GenerateResult, TriageRequest, TriageResult


class ReceiverUnavailable(RuntimeError):
    pass


class GcpIdToken:
    """Google-signed ID token for calling a private Cloud Run service, from the metadata server of the calling
    service's own identity: no key or shared secret exists anywhere. Cached until shortly before it expires."""

    URL = "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/identity"

    def __init__(self, audience: str):
        self.audience, self.token, self.expires = audience, None, 0.0

    async def __call__(self, client: httpx.AsyncClient) -> str:
        import time

        if self.token is None or time.time() > self.expires:
            r = await client.get(self.URL, params={"audience": self.audience}, headers={"Metadata-Flavor": "Google"},
                                 timeout=5.0)
            r.raise_for_status()
            self.token, self.expires = r.text, time.time() + 50 * 60  # tokens live 60 min
        return self.token


class HttpReceiver:
    """Client for the GPU inference service. Cloud Run scales that service to zero, so the first call after idle
    waits for a cold start (model load); ``timeout`` covers it.

    Errors are split by what the caller should do: network failures and 5xx raise ``ReceiverUnavailable`` (the agent
    falls back to the labelled offline rule); a 4xx means this request was refused (e.g. 409, the latent-channel
    version contract), which comes back as an unparsed answer so the guardrail tries the next channel."""

    name = "gemma-http"

    def __init__(self, base_url: str, timeout: float = 300.0, token: str | None = None, auth: str = "none"):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout)
        self._token = token
        self._id_token = GcpIdToken(self.base_url) if auth == "gcp" else None

    async def _post(self, path: str, body: dict) -> httpx.Response:
        headers = {}
        try:
            if self._id_token:
                headers["Authorization"] = f"Bearer {await self._id_token(self._client)}"
            elif self._token:
                headers["Authorization"] = f"Bearer {self._token}"
            r = await self._client.post(path, json=body, headers=headers)
        except httpx.HTTPError as exc:
            raise ReceiverUnavailable(f"{type(exc).__name__}: {exc}") from exc
        if r.status_code >= 500 or r.status_code in (401, 403):
            raise ReceiverUnavailable(f"{r.status_code}: {r.text[:200]}")
        return r

    def _refused(self, req: TriageRequest, r: httpx.Response, mode: str) -> TriageResult:
        detail = r.json().get("detail", r.text) if r.headers.get("content-type", "").startswith("application/json") \
            else r.text
        return TriageResult(channel=req.channel, tier=None, parsed=False, mode=mode, raw=f"refused ({r.status_code}): "
                                                                                       f"{str(detail)[:300]}")

    async def triage(self, req: TriageRequest) -> TriageResult:
        r = await self._post("/v1/triage", req.model_dump())
        return TriageResult(**r.json()) if r.status_code == 200 else self._refused(req, r, req.mode)

    async def explain(self, req: ExplainRequest) -> TriageResult:
        r = await self._post("/v1/explain", req.model_dump())
        return TriageResult(**r.json()) if r.status_code == 200 else self._refused(req, r, "explain")

    async def generate(self, req: GenerateRequest) -> GenerateResult | None:
        r = await self._post("/v1/generate", req.model_dump())
        return GenerateResult(**r.json()) if r.status_code == 200 else None

    async def health(self) -> dict:
        """Readiness, not just liveness: a cold GPU service answers /healthz while it is still loading the model."""
        try:
            r = await self._client.get("/readyz", timeout=5.0)
            return {"ok": r.status_code == 200, "state": "ready" if r.status_code == 200 else "loading",
                    **(r.json() if r.status_code == 200 else {})}
        except (httpx.HTTPError, ValueError) as exc:
            return {"ok": False, "state": "unreachable", "error": f"{type(exc).__name__}"}


class OfflineReceiver:
    """No LLM: answers with the rule itself. Labelled ``offline-rule`` everywhere it appears, never as Gemma."""

    name = "offline-rule"

    async def triage(self, req: TriageRequest) -> TriageResult:
        a = rule_answer(req.events)
        return TriageResult(channel=req.channel, tier=a["urgency_tier"], justification=a["justification"],
                            guideline_fact=a["referenced_guideline_fact"], source="offline-rule", latency_ms=0.0,
                            mode=req.mode)

    async def explain(self, req: ExplainRequest) -> TriageResult:
        res = await self.triage(req)
        return res.model_copy(update={"tier": req.tier, "mode": "explain"})

    async def generate(self, req: GenerateRequest) -> GenerateResult | None:
        return None  # callers fall back to their deterministic path

    async def health(self) -> dict:
        return {"ok": True, "mode": "offline-rule"}


def _gen_key(req: GenerateRequest) -> str:
    return hashlib.sha256(json.dumps(req.model_dump(), sort_keys=True).encode()).hexdigest()


class ReplayReceiver:
    """Answers recorded from real Gemma runs, keyed by what the receiver saw. Wraps a live receiver: hits are served
    from the store, misses go to the live receiver (if any) and are recorded, or to ``fallback`` when it is down."""

    def __init__(self, store: str | Path, live=None, fallback=None, record: bool = True):
        self.path = Path(store)
        self.live, self.fallback, self.record = live, fallback or OfflineReceiver(), record
        self.name = f"replay+{live.name}" if live else "replay"
        self._lock = threading.Lock()
        self._data: dict[str, dict] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    self._data[row["key"]] = row["value"]

    def __len__(self) -> int:
        return len(self._data)

    def _save(self, key: str, value: dict) -> None:
        if not self.record:
            return
        with self._lock:
            self._data[key] = value
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "value": value}) + "\n")

    async def triage(self, req: TriageRequest) -> TriageResult:
        key = "t:" + req.cache_key()
        if key in self._data:
            return TriageResult(**{**self._data[key], "source": "replay"})
        if self.live is not None:
            try:
                res = await self.live.triage(req)
                self._save(key, res.model_dump())
                return res
            except ReceiverUnavailable:
                pass
        return await self.fallback.triage(req)

    async def explain(self, req: ExplainRequest) -> TriageResult:
        key = f"e:{req.tier}:" + req.cache_key()
        if key in self._data:
            return TriageResult(**{**self._data[key], "source": "replay"})
        if self.live is not None:
            try:
                res = await self.live.explain(req)
                self._save(key, res.model_dump())
                return res
            except ReceiverUnavailable:
                pass
        return await self.fallback.explain(req)

    async def generate(self, req: GenerateRequest) -> GenerateResult | None:
        key = "g:" + _gen_key(req)
        if key in self._data:
            return GenerateResult(**{**self._data[key], "source": "replay"})
        if self.live is not None:
            try:
                res = await self.live.generate(req)
                if res is not None:
                    self._save(key, res.model_dump())
                return res
            except ReceiverUnavailable:
                pass
        return await self.fallback.generate(req)

    async def health(self) -> dict:
        live = await self.live.health() if self.live else {"ok": False}
        return {"ok": True, "replay_entries": len(self._data), "live": live}


class FaultyReceiver:
    """Evaluation only: wraps a receiver and, with probability ``p_under``, lowers its tier by one step (the
    dangerous direction), and with ``p_unparsed`` returns no answer. ``channel_bias`` scales ``p_under`` per channel.
    Seeded, so an eval run is reproducible."""

    def __init__(self, inner, p_under: float = 0.3, p_unparsed: float = 0.05, seed: int = 0,
                 channel_bias: dict[str, float] | None = None):
        self.inner, self.p_under, self.p_unparsed = inner, p_under, p_unparsed
        self.channel_bias = channel_bias or {}
        self.rng = random.Random(seed)
        self.name = f"faulty({inner.name})"
        self.injected = 0

    async def triage(self, req: TriageRequest) -> TriageResult:
        res = await self.inner.triage(req)
        u = self.rng.random()
        if u < self.p_unparsed:
            self.injected += 1
            return res.model_copy(update={"tier": None, "parsed": False, "source": "mock"})
        p = self.p_under * self.channel_bias.get(req.channel, 1.0)
        if res.tier and RANK[res.tier] > 0 and self.rng.random() < p:
            self.injected += 1
            return res.model_copy(update={"tier": TIERS[RANK[res.tier] - 1], "source": "mock"})
        return res.model_copy(update={"source": "mock"})

    async def explain(self, req: ExplainRequest) -> TriageResult:
        return await self.inner.explain(req)

    async def generate(self, req: GenerateRequest) -> GenerateResult | None:
        return await self.inner.generate(req)

    async def health(self) -> dict:
        return {"ok": True, "mode": self.name}


async def gather_limited(coros, limit: int = 4):
    sem = asyncio.Semaphore(limit)

    async def run(c):
        async with sem:
            return await c

    return await asyncio.gather(*(run(c) for c in coros))
