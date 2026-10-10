r"""The parent graph: two agents, two mailboxes, one conversation, and the policy enforcement point.

    START -> perception --(messages for reasoning)--> reasoning --(requests)--> perception -> ...
                                                          \--(done)--> END

The orchestrator owns no domain logic. It checks every message against the policy (``agent/policy.py``) before
delivering it, writes every message and verdict to the audit log, enforces the turn budget (when it runs out, the
reasoning agent is told to finish with what it has), and owns the run's lifecycle: a run ends as complete,
unreadable, failed or incomplete, and an error is never reported as a triage result.
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from ecg_agent.agent.harness import Budget, Tracer
from ecg_agent.agent.perception_agent import PerceptionAgent
from ecg_agent.agent.policy import Policy, vectors_sha256
from ecg_agent.agent.protocol import Message
from ecg_agent.agent.reasoning_agent import ReasoningAgent
from ecg_agent.observability import metrics


class OState(TypedDict, total=False):
    to_perception: list
    to_reasoning: list
    turn: int
    done: bool
    p_notices: list  # refusals of perception's messages, delivered with its next requests


class Orchestrator:
    def __init__(self, perception: PerceptionAgent, reasoning: ReasoningAgent, tracer: Tracer, policy: Policy,
                 budget: Budget | None = None, audit_path: str | Path | None = None):
        self.p, self.r, self.tracer, self.policy = perception, reasoning, tracer, policy
        self.budget = budget or Budget()
        self.audit: list[dict] = []
        self.audit_path = Path(audit_path) if audit_path else None
        self.status, self.error = "created", None
        self.started = self.ended = None
        self.overrides: list[dict] = []
        self.graph = self._build("perception")  # a run starts with perception reading the recording
        self.qa_graph = self._build("reasoning")  # a question starts with reasoning reading it

    def _build(self, entry: str):
        g = StateGraph(OState)
        g.add_node("perception", self._perception_turn)
        g.add_node("reasoning", self._reasoning_turn)
        g.add_edge(START, entry)
        g.add_conditional_edges("perception", lambda s: "reasoning" if s.get("to_reasoning") else END,
                                ["reasoning", END])
        # After reasoning: finished; or requests for perception (refusal notices ride along with its replies);
        # or only refusal notices, which reasoning reads in another turn.
        g.add_conditional_edges("reasoning", lambda s: END if s.get("done") else (
            "perception" if s.get("to_perception") else "reasoning"), ["perception", "reasoning", END])
        return g.compile()

    # ----- policy enforcement -----
    async def _route(self, msgs: list[Message]) -> tuple[list[Message], list[Message]]:
        """-> (messages delivered, refusal notices back to their senders)."""
        delivered, refusals = [], []
        for m in msgs:
            v = self.policy.check(m)
            self._audit(m, v.ok, v.reason)
            if v.ok:
                self.policy.accepted(m)
                delivered.append(m)
                await self.tracer.emit("orchestrator", "message", f"{m.sender} -> {m.recipient}: {m.intent}",
                                       message=m.brief())
                continue
            metrics.messages.labels(sender="orchestrator", intent="refused").inc()
            await self.tracer.emit("orchestrator", "guardrail", f"Refused {m.sender} -> {m.recipient}: {m.intent}",
                                   reason=v.reason, message=m.brief())
            if m.sender in ("perception", "reasoning"):
                # Tell the sender, as a failure answering its own message, so it can carry on.
                refusals.append(Message(sender="orchestrator", recipient=m.sender, performative="failure",
                                        intent=m.intent, in_reply_to=m.id,
                                        content={"error": f"refused by the orchestrator: {v.reason}",
                                                 "window_id": m.content.get("window_id"),
                                                 "channel": m.content.get("channel"),
                                                 "purpose": m.content.get("purpose", "review")}))
        return delivered, refusals

    def _audit(self, m: Message, ok: bool, reason: str) -> None:
        entry = {"t": round(time.time(), 3), "verdict": "delivered" if ok else "refused", "reason": reason or None,
                 **m.brief()}
        vec = m.content.get("payload", {}).get("vectors") if isinstance(m.content.get("payload"), dict) else None
        if vec:
            entry["vectors_sha256"] = vectors_sha256(vec)
        if m.intent == "window":
            entry["events_sha256"] = vectors_sha256([[e["classification"]["confidence"]] for e in m.content["events"]])
        self.audit.append(entry)
        if self.audit_path:
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)
            with self.audit_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")

    # ----- turns -----
    async def _perception_turn(self, s: OState) -> OState:
        out = await self.p.turn(s.get("to_perception", []))
        delivered, refusals = await self._route(out)
        # A refused perception message leaves reasoning's request open: answer it with the refusal instead.
        notices = [Message(sender="orchestrator", recipient="reasoning", performative="failure", intent=r.intent,
                           content=r.content) for r in refusals]
        carried = s.get("to_reasoning", [])  # refusal notices from reasoning's last turn
        return {"to_perception": [], "to_reasoning": carried + delivered + notices, "turn": s.get("turn", 0) + 1,
                "p_notices": s.get("p_notices", []) + refusals}

    async def _reasoning_turn(self, s: OState) -> OState:
        turn = s.get("turn", 0) + 1
        if turn >= self.budget.max_turns and self.r.report is None:
            await self.tracer.emit("orchestrator", "guardrail", "Turn budget used up; asking for the report")
            self.r.budget.max_reviews = len(self.r.findings)  # nothing more may be requested
        out = await self.r.turn(s.get("to_reasoning", []))
        delivered, refusals = await self._route(out)
        done = any(m.intent in ("done", "answer") for m in delivered) or not out
        to_p = [m for m in delivered if m.recipient == "perception"]
        notices = s.get("p_notices", []) if to_p else []  # ride along with real requests only
        return {"to_reasoning": refusals, "to_perception": notices + to_p, "turn": turn,
                "done": done or (not to_p and not refusals), "p_notices": [] if to_p else s.get("p_notices", [])}

    async def run(self) -> dict:
        t0 = time.perf_counter()
        self.status, self.started = "running", time.time()
        try:
            await self.graph.ainvoke({"turn": 0}, {"recursion_limit": 2 * self.budget.max_turns + 10})
            rep = self.r.report
            if rep is None:
                self.status = "incomplete"
            elif rep.get("status") == "failed":
                self.status = "unreadable"  # the perception agent could not read the recording
            else:
                self.status = "complete"
            return rep or {"status": "incomplete"}
        except Exception as exc:  # noqa: BLE001 - a crash is a failed run, never a triage result
            self.status, self.error = "failed", f"{type(exc).__name__}: {exc}"
            await self.tracer.emit("orchestrator", "error", "Run failed", error=self.error,
                                   trace=traceback.format_exc()[-2000:])
            return {"status": "failed", "error": self.error}
        finally:
            self.ended = time.time()
            metrics.runs.labels(status=self.status).inc()
            metrics.run_seconds.observe(time.perf_counter() - t0)

    async def ask(self, question: str) -> dict:
        """A clinician question after the run. It enters the conversation like any message (policy check, audit
        log); the reasoning agent may consult the perception agent before answering, through the same loop."""
        if self.r.report is None:
            return {"type": "refused", "text": "Questions are answered once the run has finished."}
        ask = Message(sender="clinician", recipient="reasoning", performative="request", intent="ask",
                      content={"question": question})
        delivered, _ = await self._route([ask])
        if not delivered:
            return {"type": "refused", "text": "The question could not be delivered."}
        n = len(self.audit)
        await self.qa_graph.ainvoke({"to_reasoning": delivered, "turn": 0},
                                    {"recursion_limit": 4 * self.budget.max_attempts + 12})
        answer = next((a for a in self.audit[n:] if a["intent"] == "answer" and a["verdict"] == "delivered"), None)
        return answer["content"] if answer else {"type": "error", "text": "No answer could be produced."}

    def override(self, wid: str, tier: str, clinician: str, reason: str) -> dict:
        """The clinician's control for changing a tier: a recorded human decision, not a message the agents can
        produce. It may lower a tier (that is the clinician's call), it needs a reason, and it is audited."""
        if wid not in self.r.findings:
            raise KeyError(f"{wid} has no finding")
        if tier not in ("routine", "priority", "urgent") or not clinician.strip() or not reason.strip():
            raise ValueError("an override needs a valid tier, the clinician's name and a reason")
        f = self.r.findings[wid]
        current = f["override"]["to_tier"] if f.get("override") else f["final_tier"]  # a second override chains
        if tier == current:
            raise ValueError(f"{wid} is already {tier}: an override must change the tier")
        entry = {"t": round(time.time(), 3), "verdict": "clinician_override", "window_id": wid,
                 "from_tier": current, "to_tier": tier, "clinician": clinician, "reason": reason}
        f["override"] = entry
        self.overrides.append(entry)
        self.audit.append(entry)
        if self.audit_path:
            with self.audit_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")
        return entry

    async def explain(self, wid: str) -> dict:
        """Explain one finding after the run, on request. When the reasoning agent needs the window over another
        channel, the request and the reply go through the same policy, mailboxes and audit log as the run itself."""
        first = await self.r.request_explanation(wid)
        if isinstance(first, dict):
            return first
        delivered, refusals = await self._route([first])
        if refusals:
            return {"error": refusals[0].content["error"]}
        replies, _ = await self._route(await self.p.turn(delivered))
        await self.r.turn(replies)
        return self.r.findings[wid].get("explanation") or {"error": "no explanation produced"}
