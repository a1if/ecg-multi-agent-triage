r"""The perception agent (sender): an autonomous, non-LLM agent around the CNN-LSTM.

Goal: give the reasoning agent a faithful account of the recording at the lowest communication cost.

Intake loop (its first turn), a LangGraph subgraph with a self-correcting cycle:

    acquire -> detect -> classify -> assess --(quality not good enough, strategies left)--> detect
                                         \--(good, or out of strategies: keep the best attempt)--> screen -> announce

Strategies cover the failures a single-lead pipeline meets in practice: the wrong lead, an inverted lead, a
detector threshold that misses beats. Each attempt is scored on what the agent can check without ground truth
(plausible heart rate, signal quality, unclassifiable fraction, RR regularity, abnormal fraction), and the agent
keeps the best one, saying why.

Serving loop (every later turn): answer the reasoning agent's requests. For ``request_window`` the agent picks the
channel itself (sender-side choice, the paper's question) unless the request names one, and composes the message;
``request_resend`` re-sends a window over another channel; ``request_detail`` returns a beat-by-beat listing.

The controller is a policy, not an LLM, on purpose: it runs on CPU next to the sensor, and every decision it makes
is checkable. Its decisions are traced like the reasoning agent's.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import numpy as np
from langgraph.graph import END, START, StateGraph
from scipy import signal as sps

from ecg_agent.agent.harness import Budget, Tracer
from ecg_agent.agent.protocol import Message
from ecg_agent.agent.router import choose_channel
from ecg_agent.agent.screening import Window, make_windows, rank_windows, record_summary
from ecg_agent.core.adapter import adapter_inputs
from ecg_agent.core.prompts import compact_payload, compact_prompt, filtered_prompt
from ecg_agent.core.rule import beat_tier
from ecg_agent.core.sender import Sender, SenderOutput
from ecg_agent.observability import metrics
from ecg_agent.signal import beats as B

NAME = "perception"
ADAPTER_INPUT_DIM = 35  # what the trained adapter reads


@dataclass
class Source:
    """A recording: MIT-BIH record (with reference annotations) or an uploaded multi-lead signal."""
    kind: str  # "mitdb" | "upload"
    leads: dict[str, np.ndarray]
    fs: float
    record_path: Path | None = None
    start_s: float = 0.0
    duration_s: float | None = None
    label: str = ""


def load_mitdb(records_dir: Path, record: str, start_s: float = 0.0, duration_s: float | None = 300.0) -> Source:
    import wfdb

    path = Path(records_dir) / record
    rec = wfdb.rdrecord(str(path))
    leads = {n: rec.p_signal[:, i].astype(np.float64) for i, n in enumerate(rec.sig_name)}
    return Source("mitdb", leads, rec.fs, path, start_s, duration_s, f"MIT-BIH {record}")


@dataclass
class Attempt:
    strategy: dict
    beats: B.Beats
    out: SenderOutput
    quality: dict
    score: float


class PState(TypedDict, total=False):
    attempt: int
    done: bool


# Noise: the share of a beat window's energy above 40 Hz, where an ECG has little and muscle or contact noise has a
# lot. Thresholds from docs/benchmarks.md ("Signal quality"): clean records 0.000-0.019 (median per record); at 20 dB
# SNR up to 0.041 with accuracy 0.90-1.00; at 10 dB 0.095-0.19, where the classifier misses ventricular beats
# (record 233: 3% called abnormal instead of 28%).
NOISE_SOFT, NOISE_HARD = 0.03, 0.08
UNREADABLE_MAX = 0.5  # above this share of unreadable windows, the whole recording is unreadable
_HP = sps.butter(4, 40 / (B.FS / 2), btype="high")


def beat_noise(windows: np.ndarray) -> np.ndarray:
    """(n, 360) beat windows -> (n,) share of energy above 40 Hz."""
    if not len(windows):
        return np.zeros(0)
    hp = sps.filtfilt(*_HP, windows, axis=1)
    total = np.sum((windows - windows.mean(axis=1, keepdims=True)) ** 2, axis=1)
    return np.sum(hp ** 2, axis=1) / np.maximum(total, 1e-12)


def _quality(beats: B.Beats, out: SenderOutput, source: str) -> tuple[dict, float, list[str], list[str]]:
    """-> (metrics, score, hard problems, soft problems). Hard: the classifier cannot be trusted on this input.
    Soft: it can, with less confidence (or the pattern could be real pathology), so the run goes on, flagged."""
    n = len(beats)
    dur_s = max(1e-6, len(beats.signal) / B.FS)
    labels = [e["classification"]["label"] for e in out.events]
    sqi = np.array([e["segment_metadata"]["signal_quality_index"] for e in out.events]) if n else np.zeros(1)
    rr = beats.rr_ms[1:] if n > 1 else np.array([800.0])
    noise = beat_noise(beats.windows) if n else np.ones(1)
    # Flat stretches (electrode off) contain no beats, so beat-level checks never see them: measure them directly.
    sig = beats.signal
    blocks = sig[:len(sig) // B.FS * B.FS].reshape(-1, B.FS) if len(sig) >= B.FS else np.zeros((1, B.FS))
    flat = float(np.mean(blocks.std(axis=1) < 1e-3 * max(1e-12, float(np.std(sig)) if len(sig) else 1.0)))
    q = {"beats": n, "rate_bpm": round(60.0 * n / dur_s, 1), "mean_sqi": round(float(sqi.mean()), 3),
         "q_frac": round(labels.count("Q") / max(1, n), 3),
         "abnormal_frac": round(sum(l != "N" for l in labels) / max(1, n), 3),
         "rr_cv": round(float(np.std(rr) / max(1e-6, np.mean(rr))), 3),
         "noise_median": round(float(np.median(noise)), 4),
         "noisy_beat_frac": round(float(np.mean(noise > NOISE_HARD)), 3),
         "flat_frac": round(flat, 3)}
    hard, soft = [], []
    if flat > UNREADABLE_MAX:
        hard.append(f"{flat:.0%} of the recording is flat (no signal: electrode off?)")
    elif flat > 0.05:
        soft.append(f"{flat:.0%} of the recording is flat (no signal); those stretches were not assessed")
    if n < 5:
        hard.append(f"only {n} beats found")
    if not 30 <= q["rate_bpm"] <= 220:
        hard.append(f"implausible beat rate {q['rate_bpm']} bpm")
    if q["mean_sqi"] < 0.7:
        hard.append(f"flat or clipped signal (quality index {q['mean_sqi']})")
    if q["q_frac"] > 0.1:
        hard.append(f"{q['q_frac']:.0%} unclassifiable beats")
    if q["noisy_beat_frac"] > UNREADABLE_MAX:
        hard.append(f"{q['noisy_beat_frac']:.0%} of beats too noisy to classify reliably")
    elif q["noise_median"] > NOISE_SOFT:
        soft.append(f"noise level {q['noise_median']} may reduce accuracy")
    if source == "detector" and q["abnormal_frac"] > 0.6:
        soft.append(f"{q['abnormal_frac']:.0%} abnormal beats: possible wrong lead, or real pathology")
    if source == "detector" and q["rr_cv"] > 0.6:
        soft.append(f"very irregular RR (CV {q['rr_cv']}): possible missed beats, or real arrhythmia")
    score = ((1.0 - min(1.0, q["q_frac"] * 3)) * min(1.0, q["mean_sqi"]) * (1.0 if 30 <= q["rate_bpm"] <= 220 else 0.2)
             * (1.0 - q["noisy_beat_frac"]) * (0.0 if n < 5 else 1.0))
    if source == "detector":
        score *= 1.0 - 0.5 * max(0.0, q["abnormal_frac"] - 0.6) - 0.3 * max(0.0, q["rr_cv"] - 0.6)
    return q, round(score, 4), hard, soft


class PerceptionAgent:
    name = NAME

    def __init__(self, sender: Sender, source: Source, tracer: Tracer, mode: str = "balanced",
                 budget: Budget | None = None, window_size: int = 50):
        self.sender, self.source, self.tracer, self.mode = sender, source, tracer, mode
        self.budget = budget or Budget()
        self.window_size = window_size
        self.strategies = self._strategies()
        self.attempts: list[Attempt] = []
        self.best: Attempt | None = None
        self.windows: dict[str, Window] = {}
        self.summary: dict = {}
        self.sent: dict[str, list[str]] = {}  # window id -> channels sent
        self.ready = False
        self.disabled: set[str] = set()  # channels switched off for this run after a contract refusal
        self.unreadable: list[str] = []  # set when no strategy produced a recording the sender can be trusted on
        self._intake = self._build_intake()

    # ----- strategies -----
    def _strategies(self) -> list[dict]:
        names = list(self.source.leads)
        first = "MLII" if "MLII" in names else names[0]
        order = [first] + [n for n in names if n != first]
        out = []
        if self.source.kind == "mitdb":
            out.append({"lead": first, "beats": "annotations", "invert": False,
                        "why": "reference R-peak annotations on MLII, the sender's training setting"})
        for lead in order:
            for inv in (False, True):
                out.append({"lead": lead, "beats": "detector", "invert": inv,
                            "why": f"R-peak detector on {lead}{' with polarity inverted' if inv else ''}"})
        return out

    def _beats(self, st: dict) -> B.Beats:
        s = self.source
        if st["beats"] == "annotations":
            return B.from_mitdb(s.record_path, s.start_s, s.duration_s)
        x = s.leads[st["lead"]]
        lo = int(s.start_s * s.fs)
        hi = len(x) if s.duration_s is None else min(len(x), lo + int(s.duration_s * s.fs))
        x = x[lo:hi] * (-1.0 if st["invert"] else 1.0)
        return B.from_signal(x, s.fs, {"lead": st["lead"], "inverted": st["invert"]})

    # ----- intake subgraph -----
    def _build_intake(self):
        g = StateGraph(PState)
        g.add_node("acquire", self._acquire)
        g.add_node("detect_and_classify", self._detect_and_classify)
        g.add_node("assess", self._assess)
        g.add_node("screen", self._screen)
        g.add_edge(START, "acquire")
        g.add_edge("acquire", "detect_and_classify")
        g.add_edge("detect_and_classify", "assess")
        g.add_conditional_edges("assess", lambda s: "screen" if s.get("done") else "detect_and_classify",
                                ["screen", "detect_and_classify"])
        g.add_edge("screen", END)
        return g.compile()

    async def _acquire(self, state: PState) -> PState:
        s = self.source
        await self.tracer.emit(NAME, "node", f"Acquired {s.label or s.kind}",
                               leads=list(s.leads), fs=s.fs, start_s=s.start_s, duration_s=s.duration_s,
                               plan=[x["why"] for x in self.strategies])
        return {"attempt": 0, "done": False}

    async def _detect_and_classify(self, state: PState) -> PState:
        st = self.strategies[state["attempt"]]
        empty = SenderOutput([], np.zeros((0, 32)), np.zeros((0, 5)))
        try:
            beats = self._beats(st)
            out = self.sender.process(beats.windows, beats.rr_ms) if len(beats) else empty
        except (ValueError, RuntimeError) as exc:  # one failed strategy is a failed attempt, not a failed agent
            beats = B.Beats(np.zeros(0), np.zeros(0, int), np.zeros((0, 360), np.float32), np.zeros(0),
                            meta={"source": st["beats"]})
            out = empty
            q, score, hard, soft = _quality(beats, out, st["beats"])
            hard.insert(0, f"processing failed: {type(exc).__name__}: {str(exc).splitlines()[0][:120]}")
        else:
            q, score, hard, soft = _quality(beats, out, beats.meta["source"])
        self.attempts.append(Attempt(st, beats, out, {**q, "problems": hard + soft, "hard": hard, "soft": soft},
                                     score))
        await self.tracer.emit(NAME, "tool", f"Attempt {state['attempt'] + 1}: {st['why']}",
                               strategy=st, quality=q, score=score, hard=hard, soft=soft)
        return {}

    async def _assess(self, state: PState) -> PState:
        a = self.attempts[-1]
        nxt = state["attempt"] + 1
        if not a.quality["problems"]:
            self.best = a
            await self.tracer.emit(NAME, "decision", "Quality check passed", score=a.score)
            return {"done": True}
        if nxt < len(self.strategies):
            await self.tracer.emit(NAME, "decision", "Quality check failed; trying the next strategy",
                                   problems=a.quality["problems"], next=self.strategies[nxt]["why"])
            return {"attempt": nxt}
        usable = [x for x in self.attempts if not x.quality["hard"]]
        self.best = max(usable or self.attempts, key=lambda x: x.score)
        if not usable:
            self.unreadable = self.best.quality["hard"]
            await self.tracer.emit(NAME, "decision", "Recording unreadable: stopping before any triage",
                                   reasons=self.unreadable, attempts=len(self.attempts))
        else:
            await self.tracer.emit(NAME, "decision", "Out of strategies; continuing with warnings",
                                   chosen=self.best.strategy["why"], warnings=self.best.quality["soft"])
        return {"done": True}

    async def _screen(self, state: PState) -> PState:
        if self.unreadable:
            self.ready = True
            return {}
        b, out = self.best.beats, self.best.out
        ws = make_windows(out.events, b.peaks, self.window_size)
        noise = beat_noise(b.windows)
        for w in ws:  # noise is usually local (a loose electrode): judge readability per window
            w.noise = round(float(np.median(noise[w.start:w.start + w.n])), 4)
            w.readable = w.noise <= NOISE_HARD
        self.windows = {w.id: w for w in ws}
        bad = [w.id for w in ws if not w.readable]
        warnings = list(self.best.quality["soft"])
        if bad:
            warnings.append(f"{len(bad)} of {len(ws)} windows too noisy to classify; listed for manual review")
        self.summary = {**record_summary(out.events, len(b.signal) / B.FS),
                        "strategy": self.best.strategy["why"], "quality": self.best.quality,
                        "quality_warnings": warnings, "unreadable_windows": bad}
        # Input-drift metrics: what this deployment is being given (ops/alerts.yml watches them).
        metrics.input_noise.observe(self.best.quality["noise_median"])
        metrics.windows.labels(readable="true").inc(len(ws) - len(bad))
        metrics.windows.labels(readable="false").inc(len(bad))
        for label, count in self.summary["classes"].items():
            metrics.beats.labels(label=label).inc(count)
        self.ready = True
        await self.tracer.emit(NAME, "node", f"Screened {len(out.events)} beats into {len(ws)} windows"
                               + (f", {len(bad)} unreadable" if bad else ""), summary=self.summary)
        return {}

    # ----- turns -----
    async def turn(self, inbox: list[Message]) -> list[Message]:
        if not self.ready:
            await self._intake.ainvoke({}, {"recursion_limit": 4 * len(self.strategies) + 10})
            if self.unreadable:
                return [self._msg("failure", "record_ready", {
                    "error": "recording unreadable: " + "; ".join(self.unreadable),
                    "attempts": [{"strategy": a.strategy["why"], **a.quality} for a in self.attempts]})]
            ranked = rank_windows(list(self.windows.values()))
            return [self._msg("inform", "record_ready", {
                "summary": self.summary, "candidates": [w.public() for w in ranked],
                "channels": ["compact", "filtered", "adapter"], "adapter_max_events": self.window_size})]
        # Refusal notices first, so this turn's requests are served with what they taught.
        for m in [m for m in inbox if m.sender == "orchestrator" and m.performative == "failure"]:
            await self._on_refused(m)
        out = []
        for m in [m for m in inbox if m.sender != "orchestrator"]:
            try:
                out.append(await self._serve(m))
            except (KeyError, ValueError) as exc:
                out.append(self._msg("failure", m.intent, {"error": str(exc)}, m.id))
        return out

    async def _on_refused(self, m: Message) -> None:
        """Circuit breaker: a latent message refused for its contract will be refused again (the models do not
        change mid-run), so stop offering that channel for the rest of the run instead of paying a round trip
        per window to rediscover it."""
        ch, err = m.content.get("channel"), m.content.get("error", "")
        if ch and "version contract" in err and ch not in self.disabled:
            self.disabled.add(ch)
            await self.tracer.emit(NAME, "decision", f"Stop using {ch} for this run", reason=err)

    async def _serve(self, m: Message) -> Message:
        wid = m.content.get("window_id")
        if wid not in self.windows:
            raise KeyError(f"unknown window {wid!r}")
        w = self.windows[wid]
        events = self.best.out.events[w.start:w.start + w.n]
        if not w.readable and m.intent != "request_detail":
            raise ValueError(f"{wid} is too noisy to classify reliably (noise {w.noise}); it needs manual review")
        if m.intent == "request_detail":
            rows = [{"beat": i + 1, **compact_payload(e), "rr_ms": e["signal_features"]["rr_interval_ms"]}
                    for i, e in enumerate(events)]
            await self.tracer.emit(NAME, "tool", f"Listed the beats of {wid}", window=wid)
            return self._msg("inform", "detail", {"window_id": wid, "beats": rows}, m.id)

        requested = m.content.get("channel")
        if requested in self.disabled:
            raise ValueError(f"{requested} is disabled for this run (its messages were refused)")
        if requested:
            channel, reason = requested, f"the reasoning agent asked for {requested}" + (
                f": {m.content['reason']}" if m.content.get("reason") else "")
        else:
            r = choose_channel(w.n, w.abnormal, self.mode, self.window_size)
            channel, reason = r.channel, r.reason
            if channel in self.disabled:
                channel, reason = "filtered", f"{r.channel} is disabled for this run; filtered text instead"
        if channel == "adapter" and w.n > self.window_size:
            raise ValueError(f"the adapter carries at most {self.window_size} events")
        self.sent.setdefault(wid, []).append(channel)
        metrics.channel.labels(channel=channel).inc()
        payload: dict = {"channel": channel}
        if channel == "adapter":
            # The full latent message: 32-d sender vector + 3 side inputs per beat (design 6a).
            payload["vectors"] = adapter_inputs(self.best.out.vectors[w.start:w.start + w.n], events,
                                                ADAPTER_INPUT_DIM).tolist()
            payload["sender_sha256"] = self.sender.checkpoint_sha256  # the version contract (design 6a)
        else:
            payload["prompt_chars"] = len((compact_prompt if channel == "compact" else filtered_prompt)(events))
        purpose = m.content.get("purpose", "review")
        await self.tracer.emit(NAME, "decision", f"Send {wid} over {channel}" + (" (for an explanation)"
                                                                             if purpose == "explain" else ""),
                               window=wid, channel=channel, reason=reason, resend=m.intent == "request_resend",
                               purpose=purpose)
        return self._msg("inform", "window", {
            "window_id": wid, "channel": channel, "reason": reason, "purpose": purpose, "payload": payload,
            "events": events, "screening": {"rule_tier": w.rule_tier, "abnormal": w.abnormal, "n": w.n}}, m.id)

    def _msg(self, perf: str, intent: str, content: dict, reply_to: str | None = None) -> Message:
        metrics.messages.labels(sender=NAME, intent=intent).inc()
        return Message(sender=NAME, recipient="reasoning", performative=perf, intent=intent, content=content,
                       in_reply_to=reply_to)

    # ----- for the API / UI -----
    def export(self) -> dict:
        if not self.best:
            return {}
        b, out = self.best.beats, self.best.out
        return {"peaks": b.peaks.tolist(), "labels": [e["classification"]["label"] for e in out.events],
                "confidence": [e["classification"]["confidence"] for e in out.events],
                "tiers": [beat_tier(e) for e in out.events],
                "reference": b.reference, "signal": b.signal, "windows": [w.public() for w in self.windows.values()],
                "summary": self.summary, "attempts": [{"strategy": a.strategy, "quality": a.quality,
                                                       "score": a.score} for a in self.attempts]}
