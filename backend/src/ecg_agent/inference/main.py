"""Inference service (GPU): the receiver, Gemma 4 E4B + the trained adapter, behind HTTP.

    POST /v1/triage     window -> tier (mode "decision") or tier + justification (mode "full")
    POST /v1/explain    window + decided tier -> justification
    POST /v1/generate   prompt -> text (planner, question routing, report summary)
    GET  /v1/info       model, adapter and contract versions
    GET  /healthz       process alive          GET /readyz   model loaded and warm
    GET  /metrics       Prometheus

One GPU serves one generation at a time (the engine's lock); requests queue. The service enforces the latent-channel
version contract itself (design §6a): an adapter request whose vectors come from a sender the adapter was not
trained for is refused with 409, whatever the caller checked.

Run: ``uvicorn ecg_agent.inference.main:app --port 8001`` (env: ADAPTER_CHECKPOINT, MODEL_MANIFEST, INFERENCE_TOKEN).
On Cloud Run, access is controlled by IAM (only the API service's identity may invoke it); INFERENCE_TOKEN is the
equivalent for local and docker-compose use.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from ecg_agent.observability import metrics, setup_logging
from ecg_agent.receiver.base import ExplainRequest, GenerateRequest, GenerateResult, TriageRequest, TriageResult

log = logging.getLogger("inference")
ROOT = Path(__file__).resolve().parents[3]


def default_engine():
    from ecg_agent.core.models import deployed_adapter
    from ecg_agent.receiver.gemma import GemmaEngine

    # The manifest's folder, not this file's: installed into a venv, this file is nowhere near artifacts/.
    models = Path(os.environ["MODEL_MANIFEST"]).parent if os.environ.get("MODEL_MANIFEST") else ROOT / "artifacts/models"
    return GemmaEngine(os.environ.get("ADAPTER_CHECKPOINT") or deployed_adapter(models)[0])


def create_app(engine_factory: Callable = default_engine, manifest_path: str | Path | None = None) -> FastAPI:
    manifest = json.loads(Path(manifest_path or os.environ.get("MODEL_MANIFEST",
                                                               ROOT / "artifacts/models/manifest.json")).read_text())
    token = os.environ.get("INFERENCE_TOKEN")
    state: dict = {"engine": None, "ready_s": None}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        setup_logging()
        t0 = time.perf_counter()
        state["engine"] = await asyncio.to_thread(engine_factory)  # model load + warm-up: the cold start
        state["ready_s"] = round(time.perf_counter() - t0, 1)
        log.info("receiver ready", extra={"extra_fields": {"cold_start_s": state["ready_s"]}})
        yield

    app = FastAPI(title="ECG triage: receiver (inference)", version="0.1.0", lifespan=lifespan)

    def auth(authorization: str | None = Header(default=None)) -> None:
        if token and authorization != f"Bearer {token}":
            raise HTTPException(401, "missing or wrong bearer token")

    def engine():
        if state["engine"] is None:
            raise HTTPException(503, "model is loading")
        return state["engine"]

    def contract(req: TriageRequest) -> None:
        if req.channel == "adapter" and req.sender_sha256 != manifest["adapter"]["trained_for_sender_sha256"]:
            raise HTTPException(409, "version contract: these vectors are not from the sender this adapter was "
                                     "trained for")

    async def gpu(fn, *args):
        """Run one engine call off the event loop, with the queue depth and GPU memory exported as metrics."""
        metrics.inflight.inc()
        try:
            return await asyncio.to_thread(fn, *args)
        except ValueError as exc:  # a malformed request (e.g. more events than the adapter carries)
            raise HTTPException(422, str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - every engine failure is logged and answered, never a bare crash
            oom = "out of memory" in str(exc).lower()
            log.exception("engine failure", extra={"extra_fields": {"error": type(exc).__name__, "oom": oom}})
            # Out of memory is transient (another request held the GPU): 503 tells the caller to retry or fall back.
            raise HTTPException(503 if oom else 500, f"receiver error: {type(exc).__name__}") from exc
        finally:
            metrics.inflight.dec()
            try:
                import torch

                if torch.cuda.is_available():
                    metrics.gpu_mem.set(torch.cuda.max_memory_allocated())
            except ImportError:
                pass

    @app.post("/v1/triage", response_model=TriageResult, dependencies=[Depends(auth)])
    async def triage(req: TriageRequest, eng=Depends(engine)) -> TriageResult:
        contract(req)
        res = await gpu(eng.triage, req)
        metrics.receiver_latency.labels(channel=req.channel, source="gemma").observe((res.latency_ms or 0) / 1e3)
        if res.prompt_tokens:
            metrics.receiver_tokens.labels(channel=req.channel).observe(res.prompt_tokens)
        return res

    @app.post("/v1/explain", response_model=TriageResult, dependencies=[Depends(auth)])
    async def explain(req: ExplainRequest, eng=Depends(engine)) -> TriageResult:
        contract(req)
        return await gpu(eng.explain, req, req.tier)

    @app.post("/v1/generate", response_model=GenerateResult, dependencies=[Depends(auth)])
    async def generate(req: GenerateRequest, eng=Depends(engine)) -> GenerateResult:
        return await gpu(eng.generate, req)

    @app.get("/v1/info")
    async def info() -> dict:
        return {"model": os.environ.get("GEMMA_MODEL_ID", "google/gemma-4-E4B-it"), "manifest": manifest,
                "ready": state["engine"] is not None, "cold_start_s": state["ready_s"]}

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"ok": True}

    @app.get("/readyz")
    async def readyz() -> dict:
        if state["engine"] is None:
            raise HTTPException(503, "model is loading")
        return {"ok": True, "cold_start_s": state["ready_s"]}

    @app.get("/metrics")
    async def prom(_: Request) -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()  # the model loads at startup (lifespan), not at import
