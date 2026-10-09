"""API service (CPU): runs the two agents and serves the clinician's worklist.

    GET  /v1/status                         receiver mode and whether the GPU service is warm
    GET  /v1/records, /v1/scenarios         what can be analysed (bundled records, stress scenarios)
    POST /v1/runs                           start a run on a bundled record (optionally with a stress scenario)
    POST /v1/runs/upload                    start a run on an uploaded CSV (only when ALLOW_UPLOADS is on)
    GET  /v1/runs                           the worklist: most urgent first
    GET  /v1/runs/{id}                      status, report, findings, summary
    GET  /v1/runs/{id}/events               live stream of every agent step (server-sent events)
    GET  /v1/runs/{id}/steps?after=N        the same steps as plain JSON, for clients that poll
    GET  /v1/runs/{id}/signal               min-max signal envelope + beats for a time range
    GET  /v1/runs/{id}/audit                every message and verdict
    POST /v1/runs/{id}/questions            a clinician question
    POST /v1/runs/{id}/windows/{w}/explain  explanation for one finding
    POST /v1/runs/{id}/windows/{w}/override clinician override (tier, name, reason)
    GET  /healthz, /readyz, /metrics

Run: ``uvicorn ecg_agent.api.main:app --port 8000`` (settings: ecg_agent/api/settings.py).
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from typing import Literal

import numpy as np
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from ecg_agent.api.plotting import signal_view
from ecg_agent.api.runs import RunManager
from ecg_agent.api.settings import Settings
from ecg_agent.observability import setup_logging
from ecg_agent.signal.stress import SCENARIOS

log = logging.getLogger("api")
MAX_UPLOAD_BYTES = 20 * 2**20


def jsonable(obj):
    """numpy and other non-JSON values -> plain JSON types (agent state is full of numpy scalars)."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple | set):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float | int | str | bool) or obj is None:
        return obj
    return str(obj)


class RateLimiter:
    """Sliding one-minute window per client and action. Behind Cloud Run's front end the client is the first
    address in X-Forwarded-For."""

    def __init__(self):
        self.hits: dict[tuple[str, str], deque] = defaultdict(deque)

    def check(self, request: Request, action: str, per_minute: int) -> None:
        fwd = request.headers.get("x-forwarded-for", "")
        client = fwd.split(",")[0].strip() or (request.client.host if request.client else "unknown")
        q, now = self.hits[(client, action)], time.monotonic()
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= per_minute:
            raise HTTPException(429, f"too many {action} requests; try again in a minute")
        q.append(now)


class RunRequest(BaseModel):
    record: str
    start_s: float = Field(0.0, ge=0)
    duration_s: float = Field(300.0, gt=10)
    mode: Literal["accuracy", "balanced", "throughput"] = "balanced"
    scenario: str | None = None
    max_reviews: int | None = Field(None, ge=1, le=20)


class Question(BaseModel):
    question: str = Field(min_length=2, max_length=500)


class Override(BaseModel):
    tier: Literal["routine", "priority", "urgent"]
    clinician: str = Field(min_length=2, max_length=100)
    reason: str = Field(min_length=3, max_length=500)


def parse_upload(data: bytes, fs: float) -> dict[str, np.ndarray]:
    """CSV of numbers only, one column per lead. A header row is skipped, and its names are not kept: leads are named
    lead1, lead2, ... so no text from the file can reach a prompt (design §5.2)."""
    rows = []
    for line in io.StringIO(data.decode("utf-8", errors="replace")):
        parts = [p.strip() for p in line.replace(";", ",").split(",") if p.strip()]
        try:
            rows.append([float(p) for p in parts])
        except ValueError:
            if rows:
                raise HTTPException(422, "the file must contain only numbers after an optional header row") from None
    if not rows or len({len(r) for r in rows}) != 1:
        raise HTTPException(422, "expected a CSV with the same number of numeric columns on every row")
    arr = np.asarray(rows, dtype=np.float64)
    if not np.isfinite(arr).all():
        raise HTTPException(422, "the file contains non-finite values")
    if len(arr) < 10 * fs:
        raise HTTPException(422, "the recording must be at least 10 s long")
    return {f"lead{i + 1}": arr[:, i] for i in range(arr.shape[1])}


def create_app(settings: Settings | None = None) -> FastAPI:
    s = settings or Settings.from_env()
    limiter = RateLimiter()
    state: dict = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        setup_logging()
        state["runs"] = RunManager(s)
        log.info("api ready", extra={"extra_fields": {"receiver": s.receiver, "uploads": s.allow_uploads}})
        yield
        for r in state["runs"].runs.values():
            if r.task and not r.task.done():
                r.task.cancel()

    app = FastAPI(title="ECG triage: agents API", version="0.1.0", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=list(s.cors_origins), allow_methods=["GET", "POST"],
                       allow_headers=["*"])

    def runs() -> RunManager:
        return state["runs"]

    def get_run(run_id: str, rm: RunManager = Depends(runs)):
        try:
            return rm.get(run_id)
        except KeyError:
            raise HTTPException(404, f"no run {run_id}") from None

    @app.get("/v1/status")
    async def status(rm: RunManager = Depends(runs)) -> dict:
        health = await rm.receiver.health()
        return {"receiver": s.receiver, "receiver_health": health, "planner": s.planner, "classifier": s.classifier,
                "uploads": s.allow_uploads, "max_duration_s": s.max_duration_s,
                "disclaimer": "Research prototype, not clinical advice."}

    @app.get("/v1/records")
    async def records(rm: RunManager = Depends(runs)) -> list[dict]:
        return rm.records()

    @app.get("/v1/scenarios")
    async def scenarios() -> list[str]:
        return sorted(SCENARIOS)

    @app.post("/v1/runs", status_code=202)
    async def create_run(body: RunRequest, request: Request, rm: RunManager = Depends(runs)) -> dict:
        limiter.check(request, "run", s.runs_per_minute)
        if body.scenario and body.scenario not in SCENARIOS:
            raise HTTPException(422, f"unknown scenario {body.scenario!r}")
        try:
            run = rm.create(record=body.record, start_s=body.start_s, duration_s=body.duration_s, mode=body.mode,
                            scenario=body.scenario, max_reviews=body.max_reviews)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from None
        return {"id": run.id, "status": run.status, "params": run.params}

    @app.post("/v1/runs/upload", status_code=202)
    async def upload_run(request: Request, file: UploadFile = File(...), fs: float = Form(..., ge=50, le=2000),
                         mode: str = Form("balanced"), rm: RunManager = Depends(runs)) -> dict:
        if not s.allow_uploads:
            raise HTTPException(403, "uploads are switched off in this deployment (public demo); run it locally "
                                     "with ALLOW_UPLOADS=1 to analyse your own files")
        limiter.check(request, "run", s.runs_per_minute)
        data = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "file too large (20 MB at most)")
        leads = parse_upload(data, fs)
        if mode not in ("accuracy", "balanced", "throughput"):
            raise HTTPException(422, "unknown mode")
        run = rm.create(upload=(leads, fs), duration_s=s.max_duration_s, mode=mode)
        return {"id": run.id, "status": run.status, "params": run.params}

    @app.get("/v1/runs")
    async def worklist(rm: RunManager = Depends(runs)) -> list[dict]:
        return jsonable(rm.worklist())

    @app.get("/v1/runs/{run_id}")
    async def run_detail(run=Depends(get_run)) -> dict:
        o = run.orch
        return jsonable({**run.summary(), "params": run.params, "report": o.r.report,
                         "findings": o.r.findings, "summary": o.p.summary, "windows": [
                             w.public() for w in o.p.windows.values()],
                         "answers": o.r.answered, "overrides": o.overrides, "steps": len(run.tracer.steps),
                         "attempts": [{"strategy": a.strategy["why"], **a.quality} for a in o.p.attempts]})

    @app.get("/v1/runs/{run_id}/events")
    async def events(run=Depends(get_run), after: int = -1):
        """Every step so far, then new ones as they happen, until the run ends (reconnect with ?after=<seq>)."""
        async def stream():
            ev = asyncio.Event()
            run.tracer.listeners.add(ev)
            try:
                i = after + 1
                while True:
                    steps = run.tracer.steps
                    while i < len(steps):
                        yield {"event": "step", "id": str(i), "data": json.dumps(jsonable(steps[i]))}
                        i += 1
                    if run.task is not None and run.task.done():
                        yield {"event": "end", "data": json.dumps({"status": run.status})}
                        return
                    ev.clear()
                    try:
                        await asyncio.wait_for(ev.wait(), timeout=15)
                    except TimeoutError:
                        yield {"event": "ping", "data": "{}"}  # keeps proxies from closing an idle stream
            finally:
                run.tracer.listeners.discard(ev)
        return EventSourceResponse(stream())

    @app.get("/v1/runs/{run_id}/steps")
    async def steps(run=Depends(get_run), after: int = -1, limit: int = 500) -> dict:
        """The same steps as /events, as plain JSON for clients that poll (e.g. the Streamlit frontend)."""
        s = run.tracer.steps[after + 1:after + 1 + max(1, min(limit, 2000))]
        return {"steps": jsonable(s), "status": run.status,
                "done": run.task is not None and run.task.done(), "total": len(run.tracer.steps)}

    @app.get("/v1/runs/{run_id}/signal")
    async def signal(run=Depends(get_run), start_s: float = 0.0, end_s: float = 30.0, max_points: int = 2000) -> dict:
        if end_s <= start_s:
            raise HTTPException(422, "end_s must be after start_s")
        return jsonable(signal_view(run.orch.p, start_s, end_s, max_points))

    @app.get("/v1/runs/{run_id}/audit")
    async def audit(run=Depends(get_run)) -> list[dict]:
        return jsonable(run.orch.audit)

    def finished(run):
        if run.status not in ("complete",):
            raise HTTPException(409, f"the run is {run.status}")

    @app.post("/v1/runs/{run_id}/questions")
    async def ask(body: Question, request: Request, run=Depends(get_run)) -> dict:
        limiter.check(request, "question", s.questions_per_minute)
        finished(run)
        async with run.lock:
            return jsonable(await run.orch.ask(body.question))

    @app.post("/v1/runs/{run_id}/windows/{wid}/explain")
    async def explain(wid: str, run=Depends(get_run)) -> dict:
        finished(run)
        if wid not in run.orch.r.findings:
            raise HTTPException(404, f"{wid} has no finding")
        async with run.lock:
            return jsonable(await run.orch.explain(wid))

    @app.post("/v1/runs/{run_id}/windows/{wid}/override")
    async def override(wid: str, body: Override, request: Request, run=Depends(get_run)) -> dict:
        limiter.check(request, "question", s.questions_per_minute)
        finished(run)
        try:
            return run.orch.override(wid, body.tier, body.clinician, body.reason)
        except KeyError:
            raise HTTPException(404, f"{wid} has no finding") from None

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"ok": True}

    @app.get("/readyz")
    async def readyz() -> dict:
        if "runs" not in state:
            raise HTTPException(503, "starting")
        return {"ok": True}

    @app.get("/metrics")
    async def prom() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
