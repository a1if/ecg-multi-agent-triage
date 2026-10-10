"""Runs: build the two agents and the orchestrator for one recording, run them in the background, keep the result.

The store is in memory and bounded (oldest finished runs evicted first). On Cloud Run the API therefore runs as one
instance; a shared store (Firestore) is the step to more instances, and only this module would change.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np

from ecg_agent.agent.harness import Budget, Tracer
from ecg_agent.agent.orchestrator import Orchestrator
from ecg_agent.agent.perception_agent import PerceptionAgent, Source, load_mitdb
from ecg_agent.agent.planner import GemmaPlanner, RulePlanner
from ecg_agent.agent.policy import Policy
from ecg_agent.agent.questions import GemmaClassifier, RuleClassifier
from ecg_agent.agent.reasoning_agent import ReasoningAgent
from ecg_agent.api.settings import Settings
from ecg_agent.core.rule import RANK
from ecg_agent.core.sender import Sender
from ecg_agent.observability import metrics
from ecg_agent.receiver.clients import HttpReceiver, OfflineReceiver, ReplayReceiver
from ecg_agent.signal.stress import SCENARIOS


def build_receiver(s: Settings):
    if s.receiver == "offline":
        return OfflineReceiver()
    if s.receiver == "replay":
        return ReplayReceiver(s.replay_store, live=None, record=False)
    if s.receiver == "http":
        live = HttpReceiver(s.inference_url, token=s.inference_token, auth=s.inference_auth)
        return ReplayReceiver(s.replay_store, live=live, record=True)
    raise ValueError(f"unknown RECEIVER {s.receiver!r}")


@dataclass
class Run:
    id: str
    params: dict
    orch: Orchestrator
    created: float = field(default_factory=time.time)
    task: asyncio.Task | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)  # one question / explanation at a time per run

    @property
    def status(self) -> str:
        return "queued" if self.orch.status == "created" else self.orch.status

    @property
    def tracer(self) -> Tracer:
        return self.orch.tracer

    def summary(self) -> dict:
        """One worklist row."""
        rep = self.orch.r.report or {}
        return {"id": self.id, "created": self.created, "status": self.status, "source": self.params["label"],
                "overall_tier": rep.get("overall_tier"), "needs_human_review": rep.get("needs_human_review", []),
                "quality_warnings": rep.get("quality_warnings", []), "error": rep.get("error") or self.orch.error,
                "overrides": len(self.orch.overrides)}


class RunManager:
    def __init__(self, settings: Settings):
        self.s = settings
        self.sender = Sender(settings.models_dir / "cnn_lstm_rr_seed0.pt", device="cpu")
        self.manifest_path = settings.models_dir / "manifest.json"
        self.receiver = build_receiver(settings)
        metrics.receiver_mode.labels(mode=settings.receiver).set(1)  # lets alerts tell "GPU down" from "no GPU here"
        self.runs: OrderedDict[str, Run] = OrderedDict()
        self.slots = asyncio.Semaphore(settings.max_concurrent_runs)

    def records(self) -> list[dict]:
        import wfdb

        out = []
        for hea in sorted(self.s.records_dir.glob("*.hea")):
            h = wfdb.rdheader(str(hea.with_suffix("")))
            out.append({"id": hea.stem, "database": "MIT-BIH Arrhythmia (test split)", "fs": h.fs,
                        "duration_s": round(h.sig_len / h.fs, 1), "leads": h.sig_name})
        return out

    def create(self, *, record: str | None = None, upload: tuple[dict[str, np.ndarray], float] | None = None,
               start_s: float = 0.0, duration_s: float = 300.0, mode: str = "balanced", scenario: str | None = None,
               max_reviews: int | None = None) -> Run:
        duration_s = min(float(duration_s), self.s.max_duration_s)
        if record is not None:
            if not (self.s.records_dir / f"{record}.hea").exists():
                raise KeyError(f"unknown record {record!r}")
            src = load_mitdb(self.s.records_dir, record, start_s, duration_s)
            label = f"MIT-BIH {record}, {start_s:.0f}-{start_s + duration_s:.0f} s"
        else:
            leads, fs = upload
            src = Source("upload", leads, fs, start_s=start_s, duration_s=duration_s, label="uploaded recording")
            label = "uploaded recording"
        if scenario:  # a stress scenario damages the signal; the agents must notice and cope (detector path)
            fault = SCENARIOS[scenario]
            src = Source("upload", {k: fault(v, src.fs) for k, v in src.leads.items()}, src.fs, start_s=src.start_s,
                         duration_s=src.duration_s, label=f"{label} + {scenario}")
            label = src.label
        budget = Budget(max_reviews=max_reviews or self.s.max_reviews)
        run_id = uuid.uuid4().hex[:12]
        tracer = Tracer(run_id)
        planner = GemmaPlanner(self.receiver) if self.s.planner == "gemma" else RulePlanner()
        classifier = GemmaClassifier(self.receiver) if self.s.classifier == "gemma" else RuleClassifier()
        p = PerceptionAgent(self.sender, src, tracer, mode, budget)
        r = ReasoningAgent(self.receiver, planner, tracer, mode, budget, classifier=classifier)
        audit = self.s.audit_dir / f"{run_id}.jsonl" if self.s.audit_dir else None
        orch = Orchestrator(p, r, tracer, Policy.from_manifest(self.manifest_path), budget, audit_path=audit)
        run = Run(run_id, {"label": label, "record": record, "start_s": start_s, "duration_s": duration_s,
                           "mode": mode, "scenario": scenario, "receiver": self.s.receiver}, orch)
        self.runs[run_id] = run
        self._evict()
        run.task = asyncio.create_task(self._go(run))
        return run

    async def _go(self, run: Run) -> None:
        try:
            async with self.slots:  # bounded concurrency: the sender is CPU-heavy and the GPU is shared
                await run.orch.run()
        finally:
            # Wake live streams once the task is really over (the last step is emitted just before it ends);
            # call_soon lets this coroutine finish first, so the streams see task.done() == True.
            asyncio.get_running_loop().call_soon(lambda: [ev.set() for ev in list(run.tracer.listeners)])

    def _evict(self) -> None:
        while len(self.runs) > self.s.max_stored_runs:
            oldest = next((k for k, v in self.runs.items() if v.task is None or v.task.done()), None)
            if oldest is None:
                break
            self.runs.pop(oldest)

    def get(self, run_id: str) -> Run:
        if run_id not in self.runs:
            raise KeyError(run_id)
        return self.runs[run_id]

    def worklist(self) -> list[dict]:
        """The clinician's view: most urgent first, then items needing review, then newest."""
        rows = [r.summary() for r in self.runs.values()]
        return sorted(rows, key=lambda x: (-(RANK[x["overall_tier"]] if x["overall_tier"] else
                                              (3 if x["status"] in ("unreadable", "failed") else -1)),
                                           -len(x["needs_human_review"]), -x["created"]))
