"""The inference service's HTTP contract, with a fake engine (no GPU), and the client that calls it."""
import asyncio
import json

import httpx
import pytest
from conftest import MODELS
from fastapi.testclient import TestClient

from ecg_agent.receiver.base import ExplainRequest, GenerateRequest, GenerateResult, TriageRequest, TriageResult
from ecg_agent.receiver.clients import HttpReceiver, ReceiverUnavailable

SHA = json.loads((MODELS / "manifest.json").read_text())["adapter"]["trained_for_sender_sha256"]
EVENT = {"classification": {"label": "N", "confidence": 0.99}, "clinical_flags": {"consecutive_abnormal_beats": 0},
         "segment_metadata": {"signal_quality_index": 1.0},
         "signal_features": {"heart_rate_bpm": 75.0, "rr_interval_ms": 800.0}}


class FakeEngine:
    def __init__(self, fail=False):
        self.fail, self.calls = fail, []

    def triage(self, req):
        self.calls.append(("triage", req.channel, req.mode))
        if self.fail:
            raise RuntimeError("CUDA out of memory")
        return TriageResult(channel=req.channel, tier="routine", mode=req.mode, latency_ms=12.0, prompt_tokens=553)

    def explain(self, req, tier):
        self.calls.append(("explain", req.channel, tier))
        return TriageResult(channel=req.channel, tier=tier, justification="All beats are normal.", mode="explain")

    def generate(self, req):
        return GenerateResult(text="ok")


def app(engine, monkeypatch, token=None):
    if token:
        monkeypatch.setenv("INFERENCE_TOKEN", token)
    from ecg_agent.inference.main import create_app

    return create_app(lambda: engine, MODELS / "manifest.json")


def adapter_req(sha=SHA, **kw):
    return TriageRequest(channel="adapter", events=[EVENT] * 2, vectors=[[0.0] * 35] * 2, sender_sha256=sha, **kw)


def test_triage_explain_generate_info(monkeypatch):
    eng = FakeEngine()
    with TestClient(app(eng, monkeypatch)) as c:
        assert c.get("/readyz").json()["ok"]
        r = c.post("/v1/triage", json=adapter_req(mode="decision").model_dump())
        assert r.status_code == 200 and r.json()["tier"] == "routine" and eng.calls[-1] == ("triage", "adapter", "decision")
        x = ExplainRequest(**adapter_req().model_dump(), tier="routine")
        assert c.post("/v1/explain", json=x.model_dump()).json()["justification"]
        assert c.post("/v1/generate", json=GenerateRequest(prompt="hi").model_dump()).json()["text"] == "ok"
        assert c.get("/v1/info").json()["manifest"]["adapter"]["max_events"] == 50
        assert b"inference_inflight" in c.get("/metrics").content


def test_service_enforces_the_version_contract_itself(monkeypatch):
    eng = FakeEngine()
    with TestClient(app(eng, monkeypatch)) as c:
        r = c.post("/v1/triage", json=adapter_req(sha="0" * 64).model_dump())
        assert r.status_code == 409 and "version contract" in r.json()["detail"]
        assert not eng.calls  # refused before the GPU was touched
        text = TriageRequest(channel="filtered", events=[EVENT])  # text channels carry no vectors: no contract
        assert c.post("/v1/triage", json=text.model_dump()).status_code == 200


def test_bearer_token(monkeypatch):
    with TestClient(app(FakeEngine(), monkeypatch, token="s3cret")) as c:
        body = TriageRequest(channel="filtered", events=[EVENT]).model_dump()
        assert c.post("/v1/triage", json=body).status_code == 401
        assert c.post("/v1/triage", json=body, headers={"Authorization": "Bearer s3cret"}).status_code == 200


def client_for(service) -> HttpReceiver:
    """An HttpReceiver whose HTTP goes straight into the ASGI app (no network)."""
    rec = HttpReceiver("http://inference")
    rec._client = httpx.AsyncClient(transport=httpx.ASGITransport(app=service), base_url="http://inference")
    return rec


def test_client_maps_errors_to_the_right_behaviour(monkeypatch):
    ok, broken = app(FakeEngine(), monkeypatch), app(FakeEngine(fail=True), monkeypatch)
    with TestClient(ok), TestClient(broken):  # run their startup
        async def go():
            good = client_for(ok)
            res = await good.triage(adapter_req(mode="decision"))
            assert res.tier == "routine" and res.parsed
            refused = await good.triage(adapter_req(sha="0" * 64))  # 409 -> an unparsed answer: next channel
            assert refused.tier is None and not refused.parsed and "version contract" in refused.raw
            with pytest.raises(ReceiverUnavailable):  # 500 -> unavailable: the labelled offline fallback
                await client_for(broken).triage(adapter_req())
            down = HttpReceiver("http://127.0.0.1:9", timeout=0.5)
            assert (await down.health())["state"] == "unreachable"
        asyncio.run(go())
