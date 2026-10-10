r"""The reasoning agent (receiver): Gemma plans the review, triages each window it receives, and is checked.

One turn is a LangGraph subgraph with two loops:

    observe -> triage <-> verify          (one pass per window message in the inbox)
                 \-> plan <-> plan        (re-plan when the harness rejects a decision, up to Budget.max_plan_steps)
                       \-> report         (on finish)

* ``triage`` sends the window to the receiver LLM over the channel the perception agent chose.
* ``verify`` is the guardrail: the receiver's tier may never be lower than the sender's screening tier, and an
  unparsable answer is a failure. On a failure the agent does not quietly retry: it asks the perception agent to
  re-send the window over the next channel (filtered text first, the most accurate). After Budget.max_attempts
  channels it settles on the screening tier and marks the window for human review.
* ``plan`` asks the planner (Gemma, with a deterministic fallback) for the next tool call; the harness validates it
  (known window, not yet reviewed, budget left) and refuses to finish while an urgent window is unreviewed.
* ``report`` writes the structured report and asks Gemma for a short narrative over it (template if unavailable).

A turn ends when the agent has sent a message: a request to the perception agent, or ``done``.

Reviews stop at the tier (decide early). Justifications are written later, on request (``request_explanation``):
by the rule for guardrail overrides, by the receiver on text channels, and for adapter decisions over filtered text
after an independent decision there agrees (``explain_window``). Every LLM explanation passes the grounding check.
"""
from __future__ import annotations

import json
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from ecg_agent.agent import grounding
from ecg_agent.agent.harness import Budget, Tracer, call_tool
from ecg_agent.agent.planner import Decision, RulePlanner
from ecg_agent.agent.protocol import Message
from ecg_agent.agent.questions import REFUSALS, REFUSED_TYPES, RuleClassifier, safety_screen
from ecg_agent.agent.router import next_channel
from ecg_agent.core.rule import RANK, TIERS, window_tier
from ecg_agent.observability import metrics
from ecg_agent.receiver.base import ExplainRequest, GenerateRequest, TriageRequest, TriageResult
from ecg_agent.receiver.clients import OfflineReceiver, ReceiverUnavailable

NAME = "reasoning"

REPORT_SYSTEM = ("You write the summary of an automated ECG triage run for a clinician. Use only the facts in the "
                 "JSON you are given; do not add diagnoses, numbers or advice that are not there. Plain sentences, "
                 "no lists, at most four sentences. End with: Research prototype, not clinical advice.")


SYSTEM_ANSWER = (
    "Two agents review the recording. The perception agent, a CNN-LSTM on CPU, labels every beat and flags urgent ones "
    "with a fixed rule; the reasoning agent, Gemma 4 E4B, decides which parts to review and triages them over one of "
    "three channels (every beat as text, only abnormal beats as text, or compact vectors). A guardrail never lets "
    "its answer fall below the rule's tier, and a clinician signs off. Research prototype, not clinical advice.")


# How a clinician should read each answer source (the internal names stay in the data and the audit trail).
SOURCE_WORDS = {"gemma": "Gemma", "replay": "Gemma (recorded answer)", "offline-rule": "the triage rule (no LLM)",
                "rule": "the triage rule", "mock": "a simulated receiver"}


class RState(TypedDict, total=False):
    queue: list  # window messages still to triage this turn
    xqueue: list  # window messages sent for an explanation
    current: dict  # the window message being triaged, with its result
    plan_steps: int
    finish: bool


class ReasoningAgent:
    name = NAME

    def __init__(self, receiver, planner, tracer: Tracer, mode: str = "balanced", budget: Budget | None = None,
                 classifier=None):
        self.receiver, self.planner, self.tracer, self.mode = receiver, planner, tracer, mode
        self.classifier = classifier or RuleClassifier()
        self.q: dict | None = None  # the clinician question being answered
        self.answered: list[dict] = []
        self.q_reviews = 0
        self.budget = budget or Budget()
        self.summary: dict = {}
        self.candidates: list[dict] = []
        self.findings: dict[str, dict] = {}  # window id -> final finding
        self.attempts: dict[str, list[dict]] = {}  # window id -> every receiver answer
        self.history: list[str] = []  # what the planner sees of its own past
        self.details: dict[str, list[dict]] = {}
        self.accepted: dict[str, Message] = {}  # window id -> the window message whose answer was accepted
        self.detail_msgs: dict[str, str] = {}  # window id -> id of the detail message (for citations)
        self.outbox: list[Message] = []
        self.report: dict | None = None
        self.failed: str | None = None
        self._graph = self._build()

    # ----- graph -----
    def _build(self):
        g = StateGraph(RState)
        for n in ("observe", "explain_window", "triage", "verify", "plan", "report"):
            g.add_node(n, getattr(self, f"_{n}"))
        g.add_edge(START, "observe")
        g.add_conditional_edges("observe", self._after_observe, ["explain_window", "triage", "plan", END])
        g.add_conditional_edges("explain_window", self._after_observe, ["triage", "plan", END])
        g.add_edge("triage", "verify")
        g.add_conditional_edges("verify", lambda s: "triage" if s.get("queue") else (
            "plan" if not self.outbox and self.report is None else END), ["triage", "plan", END])
        g.add_conditional_edges("plan", self._after_plan, ["plan", "report", END])
        g.add_edge("report", END)
        return g.compile()

    async def _after_refusal(self, m: Message) -> None:
        """A window the orchestrator refused (e.g. a latent message that breaks the version contract) counts as a
        failed attempt on that channel: ask for the next channel, or settle on the screening tier for human review."""
        wid, channel = m.content.get("window_id"), m.content.get("channel")
        if m.intent != "window" or not wid or wid in self.findings or m.content.get("purpose") == "explain":
            return
        tries = self.attempts.setdefault(wid, [])
        tries.append({"channel": channel, "tier": None, "verdict": "refused", "source": "orchestrator",
                      "prompt_tokens": None, "latency_ms": None, "ttft_ms": None, "justification": "",
                      "guideline_fact": "", "reason": m.content.get("error")})
        metrics.guardrail.labels(channel=channel or "unknown", verdict="refused").inc()
        nxt = next_channel([t["channel"] for t in tries])
        if nxt and len(tries) < self.budget.max_attempts:
            self._send("request_resend", {"window_id": wid, "channel": nxt,
                                          "reason": f"{channel} message was refused by the orchestrator"})
            return
        rule = next((c["rule_tier"] for c in self.candidates if c["id"] == wid), "urgent")  # unknown -> safest
        self.findings[wid] = {"window_id": wid, "final_tier": rule, "screening_tier": rule,
                              "resolution": "guardrail_override", "needs_human_review": True, "attempts": tries,
                              "n": None, "abnormal": None}

    def _after_observe(self, s: RState) -> str:
        if self.failed:
            return END
        if self.outbox and not s.get("queue") and not s.get("xqueue"):
            return END  # a resend after a refusal is already on its way
        if s.get("xqueue"):
            return "explain_window"
        if s.get("queue"):
            return "triage"
        return END if self.report is not None else "plan"  # after the report, turns only serve explanations

    def _after_plan(self, s: RState) -> str:
        if s.get("finish"):
            return "report"
        if self.outbox:
            return END
        return "plan" if s.get("plan_steps", 0) < self.budget.max_plan_steps else "report"

    async def turn(self, inbox: list[Message]) -> list[Message]:
        self.outbox = []
        await self._graph.ainvoke({"queue": [m for m in inbox], "plan_steps": 0},
                                  {"recursion_limit": 8 * self.budget.max_plan_steps + 4 * len(inbox) + 20})
        return self.outbox

    # ----- nodes -----
    async def _observe(self, s: RState) -> RState:
        windows, for_explaining = [], []
        for m in s.get("queue", []):
            if m.intent == "record_ready" and m.performative == "failure":
                self.failed = m.content.get("error", "perception failed")
                self.report = {"status": "failed", "error": self.failed, "narrative": f"No triage: {self.failed}."}
                self._send("done", {"report": self.report})
            elif m.intent == "record_ready":
                self.summary, self.candidates = m.content["summary"], m.content["candidates"]
                await self.tracer.emit(NAME, "node", "Received the record summary",
                                       beats=self.summary["beats"], windows=len(self.candidates))
            elif m.intent == "ask":
                await self._start_question(m)
            elif m.intent == "detail":
                wid = m.content["window_id"]
                self.details[wid] = m.content["beats"]
                self.detail_msgs[wid] = m.id
                listed = [b for b in m.content["beats"] if b["label"] != "N"]
                self.history.append(f"request_detail {wid} -> {len(listed)} non-normal beats: "
                                    + json.dumps(listed[:12]))
                await self._maybe_answer()
            elif m.performative == "failure":
                self.history.append(f"{m.intent} failed: {m.content.get('error')}")
                await self.tracer.emit(NAME, "observation", f"{m.sender.capitalize()} refused a message",
                                       **m.content)
                await self._after_refusal(m)
            elif m.intent == "window":
                (for_explaining if m.content.get("purpose") == "explain" else windows).append(m)
        return {"queue": windows, "xqueue": for_explaining}

    async def _triage(self, s: RState) -> RState:
        m, rest = s["queue"][0], s["queue"][1:]
        c = m.content
        # Decide early: the guardrail needs only the tier; the justification is written later, on demand.
        req = TriageRequest(channel=c["channel"], events=c["events"], vectors=c["payload"].get("vectors"),
                            mode="decision", sender_sha256=c["payload"].get("sender_sha256"))
        try:
            res = await call_tool(self.tracer, NAME, "triage", lambda: self.receiver.triage(req),
                                  self.budget.receiver_timeout_s, window=c["window_id"])
        except (ReceiverUnavailable, TimeoutError) as exc:
            await self.tracer.emit(NAME, "observation", "Receiver unavailable; answering with the rule, labelled",
                                   error=str(exc)[:200])
            res = await OfflineReceiver().triage(req)
        if res.prompt_tokens:
            metrics.receiver_tokens.labels(channel=res.channel).observe(res.prompt_tokens)
        if res.latency_ms is not None:
            metrics.receiver_latency.labels(channel=res.channel, source=res.source).observe(res.latency_ms / 1e3)
        await self.tracer.emit(NAME, "tool", f"Triaged {c['window_id']} over {c['channel']}: {res.tier or 'no answer'}",
                               window=c["window_id"], channel=c["channel"], tier=res.tier, source=res.source,
                               justification=res.justification, prompt_tokens=res.prompt_tokens,
                               latency_ms=res.latency_ms, ttft_ms=res.ttft_ms)
        return {"queue": rest, "current": {"msg": m, "res": res}}

    async def _verify(self, s: RState) -> RState:
        m: Message = s["current"]["msg"]
        res: TriageResult = s["current"]["res"]
        wid, screen = m.content["window_id"], m.content["screening"]
        rule = screen["rule_tier"]
        if res.tier is None:
            verdict = "unparsed"
        elif RANK[res.tier] < RANK[rule]:
            verdict = "under_triage"
        elif RANK[res.tier] > RANK[rule]:
            verdict = "over_triage"
        else:
            verdict = "agree"
        metrics.guardrail.labels(channel=res.channel, verdict=verdict).inc()
        tries = self.attempts.setdefault(wid, [])
        tries.append({"channel": res.channel, "tier": res.tier, "verdict": verdict, "source": res.source,
                      "prompt_tokens": res.prompt_tokens, "latency_ms": res.latency_ms, "ttft_ms": res.ttft_ms,
                      "justification": res.justification, "guideline_fact": res.guideline_fact,
                      "reason": m.content.get("reason")})

        if verdict in ("agree", "over_triage"):
            final, resolution = res.tier, "accepted" if verdict == "agree" else "accepted_more_cautious"
        else:
            nxt = next_channel([t["channel"] for t in tries])
            if nxt and len(tries) < self.budget.max_attempts:
                why = ("the answer did not parse" if verdict == "unparsed"
                       else f"{res.channel} answered {res.tier}, below the screening tier {rule}")
                await self.tracer.emit(NAME, "guardrail", f"Rejected {wid}: {why}; asking for {nxt}",
                                       window=wid, verdict=verdict, next_channel=nxt)
                self._send("request_resend", {"window_id": wid, "channel": nxt, "reason": why}, m.id)
                self.history.append(f"request_window {wid} over {res.channel} -> {res.tier}; guardrail rejected "
                                    f"({verdict}), asked for {nxt}")
                return {}
            final, resolution = rule, "guardrail_override"
        self.accepted[wid] = m
        self.findings[wid] = {"window_id": wid, "final_tier": final, "screening_tier": rule, "resolution": resolution,
                              "needs_human_review": resolution == "guardrail_override",
                              "attempts": tries, "n": screen["n"], "abnormal": screen["abnormal"]}
        if self.q and self.q.get("waiting") == ("finding", wid):
            self.findings[wid]["origin"] = "question"
        self.history.append(f"request_window {wid} -> final {final} ({resolution}, {len(tries)} attempt(s))")
        await self.tracer.emit(NAME, "guardrail", f"{wid}: {final} ({resolution.replace('_', ' ')})", window=wid,
                               verdict=verdict, final_tier=final, resolution=resolution)
        await self._maybe_answer()
        return {}

    def _view(self) -> dict:
        return {"summary": {k: v for k, v in self.summary.items() if k != "quality"}, "candidates": self.candidates,
                "reviewed": list(self.findings), "budget": self.budget.max_reviews, "mode": self.mode,
                "history": self.history}

    async def _plan(self, s: RState) -> RState:
        steps = s.get("plan_steps", 0) + 1
        view = self._view()
        # Last planning step of the turn: use the deterministic planner so the turn always makes progress.
        planner = self.planner if steps < self.budget.max_plan_steps else RulePlanner()
        d: Decision = await call_tool(self.tracer, NAME, "plan", lambda: planner.decide(view))
        problem = self._check(d, view)
        if problem and problem.startswith("override:"):
            # The harness overrules the planner (coverage or budget) rather than letting it loop.
            forced = await RulePlanner().decide(view) if "coverage" in problem else Decision("finish")
            await self.tracer.emit(NAME, "guardrail", problem[9:], planner_wanted=d.tool, args=d.args,
                                   instead=forced.tool, instead_args=forced.args)
            d = Decision(forced.tool, forced.args, forced.thought, "harness")
            problem = None
        await self.tracer.emit(NAME, "plan", f"{d.tool} {d.args.get('window_id', '')}".strip(), thought=d.thought,
                               planner=d.source, tool=d.tool, args=d.args, rejected=problem)
        if problem:
            self.history.append(f"{d.tool} {json.dumps(d.args)} rejected by the harness: {problem}")
            return {"plan_steps": steps}
        if d.tool == "finish":
            return {"plan_steps": steps, "finish": True}
        content = {"window_id": d.args["window_id"]}
        if d.tool == "request_window" and d.args.get("channel"):
            content["channel"] = d.args["channel"]
        self._send(d.tool, content)
        return {"plan_steps": steps}

    def _check(self, d: Decision, view: dict) -> str | None:
        ids = {w["id"] for w in self.candidates}
        unreviewed_urgent = [w["id"] for w in self.candidates
                             if w["rule_tier"] == "urgent" and w["id"] not in self.findings and w.get("readable", True)]
        unreadable = {w["id"] for w in self.candidates if not w.get("readable", True)}
        budget_left = len(self.findings) < self.budget.max_reviews
        if d.tool == "finish":
            if unreviewed_urgent and budget_left:
                return f"override:coverage: cannot finish while {unreviewed_urgent[0]} (urgent by screening) is unreviewed"
            return None
        wid = d.args.get("window_id")
        if wid not in ids:
            return f"unknown window_id {wid!r}"
        if d.tool == "request_window":
            if wid in unreadable:
                return f"{wid} is unreadable (too noisy); it is listed for manual review, not triaged"
            if wid in self.findings:
                return f"{wid} was already reviewed"
            if not budget_left:
                return "override:budget: review budget used up"
            if d.args.get("channel") not in (None, "compact", "filtered", "adapter"):
                return f"unknown channel {d.args.get('channel')!r}"
        if d.tool == "request_detail" and wid in self.details:
            return f"the beats of {wid} were already listed"
        return None

    async def _report(self, s: RState) -> RState:
        findings = sorted(self.findings.values(), key=lambda f: (-RANK[f["final_tier"]], f["window_id"]))
        reviewed = set(self.findings)
        unreadable = [w["id"] for w in self.candidates if not w.get("readable", True)]
        unreviewed = [w for w in self.candidates if w["id"] not in reviewed and w["id"] not in unreadable]
        # Unreadable windows' rule tiers come from labels the sender cannot be trusted on: they do not set the
        # overall tier; they are listed for manual review instead (and a recording mostly unreadable stops earlier).
        overall = max([f["final_tier"] for f in findings] + [w["rule_tier"] for w in self.candidates
                                                             if w.get("readable", True)],
                      key=RANK.__getitem__, default="routine")
        all_tries = [t for f in findings for t in f["attempts"]]
        report = {
            "status": "complete", "overall_tier": overall, "summary": self.summary,
            "findings": findings,
            "unreviewed": {t: sum(w["rule_tier"] == t for w in unreviewed) for t in TIERS},
            "needs_human_review": [f["window_id"] for f in findings if f["needs_human_review"]] + unreadable,
            "unreadable_windows": unreadable,
            "quality_warnings": self.summary.get("quality_warnings", []),
            "receiver": {
                "calls": len(all_tries),
                "agreement": round(sum(t["verdict"] == "agree" for t in all_tries) / max(1, len(all_tries)), 3),
                "by_channel": {c: {"calls": len(ts), "agree": sum(t["verdict"] == "agree" for t in ts),
                                   "mean_prompt_tokens": round(sum(t["prompt_tokens"] or 0 for t in ts) / len(ts))}
                               for c in ("compact", "filtered", "adapter")
                               if (ts := [t for t in all_tries if t["channel"] == c])},
                "sources": sorted({t["source"] for t in all_tries}),
            },
        }
        facts = {"overall_tier": overall, "beats": self.summary.get("beats"),
                 "duration_s": self.summary.get("duration_s"), "median_heart_rate": self.summary.get("hr_median"),
                 "beat_classes": self.summary.get("classes"),
                 "longest_abnormal_run": self.summary.get("longest_abnormal_run"),
                 "windows_reviewed": [{"window": f["window_id"], "tier": f["final_tier"],
                                       "resolution": f["resolution"]} for f in findings],
                 "windows_needing_human_review": report["needs_human_review"],
                 "unreadable_windows": unreadable, "quality_warnings": report["quality_warnings"]}
        narrative, source = None, "template"
        try:
            res = await call_tool(self.tracer, NAME, "write_report", lambda: self.receiver.generate(GenerateRequest(
                prompt="Facts:\n" + json.dumps(facts, indent=1) + "\n\nSummary:", system=REPORT_SYSTEM,
                max_new_tokens=180)), self.budget.receiver_timeout_s)
            if res is not None and res.text.strip():
                problems = grounding.check_narrative(res.text, facts)
                if problems:  # an ungrounded summary is not shown; the template below is, and the reason is kept
                    await self.tracer.emit(NAME, "guardrail", "Summary failed the grounding check; using the template",
                                           rejected=res.text, problems=problems)
                    source = f"template (LLM summary rejected: {'; '.join(problems)})"
                else:
                    narrative, source = res.text.strip(), res.source
        except Exception:  # noqa: BLE001 - the report must be written whatever the LLM does
            pass
        if narrative is None:
            overridden = [f["window_id"] for f in findings if f["needs_human_review"]]
            parts = [f"{facts['beats']} beats over {facts['duration_s']} s were screened; the most urgent tier found "
                     f"is {overall}. {len(findings)} window(s) were reviewed by the receiver."]
            if overridden:
                parts.append(f"{len(overridden)} need human review because the receiver's answer could not be "
                             "confirmed.")
            if unreadable:
                parts.append(f"{len(unreadable)} were too noisy to classify and need manual review.")
            if report["quality_warnings"]:
                parts.append("Quality warnings: " + "; ".join(report["quality_warnings"]) + ".")
            narrative = " ".join(parts + ["Research prototype, not clinical advice."])
        report["narrative"], report["narrative_source"] = narrative, source
        self.report = report
        await self.tracer.emit(NAME, "node", f"Report: overall {overall}", overall_tier=overall,
                               reviewed=len(findings), human_review=report["needs_human_review"])
        self._send("done", {"report": report})
        return {}

    # ----- clinician questions (design §6c) -----
    async def _start_question(self, m: Message) -> None:
        """Screen, classify and dispatch one question. Answers that need evidence from the perception agent are
        composed when it arrives (``_maybe_answer``); everything else is answered in this turn."""
        question = str(m.content["question"])[:1000]
        self.q = {"msg": m, "question": question, "waiting": None}
        await self.tracer.emit(NAME, "node", "Question from the clinician", question=question)
        if len(self.answered) >= self.budget.max_questions:
            return self._answer("refused", REFUSALS["budget"])
        refusal = safety_screen(question)
        if refusal:
            await self.tracer.emit(NAME, "guardrail", f"Question refused before any LLM saw it ({refusal})")
            return self._answer("refused", REFUSALS[refusal], data={"refusal": refusal})
        view = {"candidates": self.candidates, "reviewed": set(self.findings)}
        intent = await call_tool(self.tracer, NAME, "classify", lambda: self.classifier.classify(question, view))
        await self.tracer.emit(NAME, "decision", f"Question type: {intent.type}" +
                               (f" about {intent.window_id}" if intent.window_id else ""),
                               classifier=intent.source, notes=intent.notes)
        self.q["intent"] = intent
        if intent.type in REFUSED_TYPES:  # the second layer caught what the patterns missed
            key = REFUSED_TYPES[intent.type]
            await self.tracer.emit(NAME, "guardrail", f"Question refused by the classifier ({key})")
            return self._answer("refused", REFUSALS[key], data={"refusal": key, "layer": "classifier"})
        wid = intent.window_id
        if intent.type == "explain":
            first = await self.request_explanation(wid)
            if isinstance(first, Message):
                self.outbox.append(first)
                self.q["waiting"] = ("explain", wid)
            else:
                self._answer_explain(wid)
        elif intent.type == "evidence":
            if wid in self.details:
                return self._answer_evidence(wid)
            self._send("request_detail", {"window_id": wid, "purpose": "question"})
            self.q["waiting"] = ("detail", wid)
        elif intent.type == "review":
            if any(w["id"] == wid and not w.get("readable", True) for w in self.candidates):
                return self._answer("review", f"{wid} is too noisy to classify reliably, so it was not triaged; it "
                                              "is listed for manual review.", wid)
            if self.q_reviews >= self.budget.max_question_reviews:
                return self._answer("refused", f"Questions have already had {self.q_reviews} extra windows reviewed "
                                               "in this run, the limit; ask about a reviewed window instead.")
            self.q_reviews += 1
            self._send("request_window", {"window_id": wid, "purpose": "review"})
            self.q["waiting"] = ("finding", wid)
        elif intent.type == "compare":
            self._answer_compare(wid)
        elif intent.type == "system":
            self._answer("system", SYSTEM_ANSWER)
        else:
            reviewed = ", ".join(sorted(self.findings)) or "none yet"
            self._answer("unclear", "I can explain a reviewed window, show its beats, review another part of the "
                                    "recording, or compare the channels. Name a window (e.g. w003) or a time (e.g. "
                                    f"1:32). Reviewed windows: {reviewed}.")

    async def _maybe_answer(self) -> None:
        if not self.q or not self.q.get("waiting"):
            return
        kind, wid = self.q["waiting"]
        f = self.findings.get(wid, {})
        if kind == "detail" and wid in self.details:
            self._answer_evidence(wid)
        elif kind == "explain" and "explanation" in f:
            self._answer_explain(wid)
        elif kind == "finding" and f:
            if self.report is not None:  # a question can make the report more urgent, never less
                self.report.setdefault("added_by_questions", []).append(wid)
                if RANK[f["final_tier"]] > RANK[self.report["overall_tier"]]:
                    self.report["overall_tier"] = f["final_tier"]
            tries = f["attempts"]
            self._answer("review", f"Reviewed {wid}: {f['final_tier']} ({f['resolution'].replace('_', ' ')}); the "
                                   f"screening said {f['screening_tier']}. Channel(s): "
                                   f"{', '.join(str(t['channel']) for t in tries)}. Ask why for the explanation.",
                         wid, self._cite(wid), data={"finding": {k: v for k, v in f.items() if k != "attempts"}})

    def _answer_explain(self, wid: str) -> None:
        f = self.findings[wid]
        e = f["explanation"]
        note = f" ({e['note']})" if e.get("note") else ""
        source, _, channel = e["explained_by"].partition(" over ")
        who = SOURCE_WORDS.get(source, source) + (f", over {channel} text" if channel else "")
        text = (f"{wid} is {f['final_tier']} ({f['resolution'].replace('_', ' ')}). {e['justification']} "
                f"Guideline: {e['guideline_fact']} Explained by {who}{note}.")
        self._answer("explain", text, wid, self._cite(wid), data={"explanation": e})

    def _answer_evidence(self, wid: str) -> None:
        beats = self.details[wid]
        w = next(c for c in self.candidates if c["id"] == wid)
        odd = [b for b in beats if b["label"] != "N"]
        listed = "; ".join(f"beat {b['beat']} {b['label']} (confidence {b['confidence']:.2f}, run "
                           f"{b['consecutive_abnormal_beats']})" for b in odd[:15])
        more = f"; and {len(odd) - 15} more" if len(odd) > 15 else ""
        text = (f"{wid} covers {w['t_start_s']:.0f}-{w['t_end_s']:.0f} s, {len(beats)} beats, {len(odd)} non-normal"
                + (f": {listed}{more}." if odd else "; all normal."))
        self._answer("evidence", text, wid, [self.detail_msgs[wid]], data={"beats": beats})

    def _answer_compare(self, wid: str | None) -> None:
        rows = [f for f in self.findings.values() if wid in (None, f["window_id"])]
        parts, rejected = [], 0
        for f in sorted(rows, key=lambda f: f["window_id"]):
            tries = ", ".join(f"{t['channel']} {t['tier'] or t['verdict']}" for t in f["attempts"])
            rejected += sum(t["verdict"] != "agree" for t in f["attempts"])
            parts.append(f"{f['window_id']}: {tries} -> {f['final_tier']}")
        text = (f"{len(rows)} window(s); {rejected} receiver answer(s) were rejected or refused. " + "; ".join(parts)
                if rows else "No window has been reviewed yet.")
        cites = [c for f in rows for c in self._cite(f["window_id"])]
        self._answer("compare", text, wid, cites)

    def _cite(self, wid: str) -> list[str]:
        m = self.accepted.get(wid)
        return [m.id] if m else []

    def _answer(self, kind: str, text: str, wid: str | None = None, citations: list[str] | None = None,
                data: dict | None = None) -> None:
        q = self.q
        content = {"type": kind, "text": text, "window_id": wid, "citations": citations or [], "data": data or {},
                   "question": q["question"]}
        self.answered.append(content)
        metrics.messages.labels(sender=NAME, intent="answer").inc()
        self.outbox.append(Message(sender=NAME, recipient="clinician", performative="inform", intent="answer",
                                   content=content, in_reply_to=q["msg"].id))
        self.q = None

    # ----- explain later (design §6d, "explanation channel") -----
    async def request_explanation(self, wid: str) -> dict | Message:
        """Start explaining one finding. Returns the explanation when it can be written now, or a request to the
        perception agent when the window must first be fetched over a readable channel.

        * guardrail override        -> the rule's wording: an LLM must not justify a tier it disagreed with;
        * decided over text         -> the receiver explains on that channel, resuming after the tier it chose;
        * decided over the adapter  -> adapter explanations are often degenerate, so ask for the window over
                                       filtered text and decide there first (see ``_explain_window``)."""
        if wid not in self.findings:
            raise KeyError(f"{wid} has no finding to explain")
        f = self.findings[wid]
        if "explanation" in f:
            return f["explanation"]
        if wid not in self.accepted:  # every attempt was refused: there is no receiver answer to explain
            f["explanation"] = {"justification": "No receiver answer could be obtained for this window; its tier is "
                                                 "the screening rule's, pending human review.",
                                "guideline_fact": "", "explained_by": "rule", "note": "all channels refused",
                                "latency_ms": None, "generated_tokens": None}
            return f["explanation"]
        m = self.accepted[wid]
        if f["resolution"] == "guardrail_override":
            return await self._by_rule(wid, m, "the guardrail set this tier, not the receiver")
        if m.content["channel"] != "adapter":
            return await self._by_receiver(wid, m, f["final_tier"])
        metrics.messages.labels(sender=NAME, intent="request_window").inc()
        return Message(sender=NAME, recipient="perception", performative="request", intent="request_window",
                       content={"window_id": wid, "channel": "filtered", "purpose": "explain",
                                "reason": "adapter decisions are explained over readable text"})

    async def _explain_window(self, s: RState) -> RState:
        """A window re-sent over filtered text for an explanation: decide there independently; explain only if the
        tier agrees with the finding, otherwise use the rule's wording and record the disagreement."""
        for m in s.get("xqueue", []):
            wid = m.content["window_id"]
            final = self.findings[wid]["final_tier"]
            c = m.content
            req = TriageRequest(channel=c["channel"], events=c["events"], mode="decision")
            d = await call_tool(self.tracer, NAME, "triage", lambda req=req: self.receiver.triage(req),
                                self.budget.receiver_timeout_s, window=wid, purpose="explain")
            await self.tracer.emit(NAME, "tool", f"Independent decision on {wid} over {c['channel']}: {d.tier}",
                                   window=wid, tier=d.tier, finding_tier=final, purpose="explain")
            if d.tier == final:
                await self._by_receiver(wid, m, final)
            else:
                await self._by_rule(wid, m, f"{c['channel']} text decided {d.tier}, not {final}")
        await self._maybe_answer()
        return {"xqueue": []}

    async def _by_receiver(self, wid: str, m: Message, tier: str) -> dict:
        c = m.content
        req = ExplainRequest(channel=c["channel"], events=c["events"], vectors=c["payload"].get("vectors"), tier=tier,
                             sender_sha256=c["payload"].get("sender_sha256"))
        res = await call_tool(self.tracer, NAME, "explain", lambda: self.receiver.explain(req),
                              self.budget.receiver_timeout_s, window=wid)
        bad = grounding.check(f"{res.justification} {res.guideline_fact}", c["events"])
        if not res.parsed or bad:
            why = "the explanation did not parse" if not res.parsed else f"numbers not in the evidence: {bad[:5]}"
            await self.tracer.emit(NAME, "guardrail", f"Explanation of {wid} failed the grounding check", window=wid,
                                   rejected=res.justification, reason=why)
            return await self._by_rule(wid, m, why)
        return await self._store(wid, res, f"{res.source} over {c['channel']}", None)

    async def _by_rule(self, wid: str, m: Message, note: str) -> dict:
        c = m.content
        final = self.findings[wid]["final_tier"]
        res = await OfflineReceiver().explain(ExplainRequest(channel=c["channel"], events=c["events"], tier=final))
        rule = window_tier(c["events"])
        if rule != final:  # the receiver was more cautious than the rule; the rule's words describe the rule's tier
            note = f"{note}; the rule's own tier is {rule}, the receiver's more cautious {final} was kept"
        return await self._store(wid, res, "rule", note)

    async def _store(self, wid: str, res: TriageResult, by: str, note: str | None) -> dict:
        exp = {"justification": res.justification, "guideline_fact": res.guideline_fact, "explained_by": by,
               "note": note, "latency_ms": res.latency_ms, "generated_tokens": res.generated_tokens}
        self.findings[wid]["explanation"] = exp
        await self.tracer.emit(NAME, "tool", f"Explained {wid} ({by})", window=wid, **exp)
        return exp

    def _send(self, intent: str, content: dict, reply_to: str | None = None) -> None:
        metrics.messages.labels(sender=NAME, intent=intent).inc()
        recipient = "orchestrator" if intent == "done" else "perception"
        self.outbox.append(Message(sender=NAME, recipient=recipient, performative="inform" if intent == "done"
                                   else "request", intent=intent, content=content, in_reply_to=reply_to))
