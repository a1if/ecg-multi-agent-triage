"""Planners pick the reasoning agent's next tool call.

``GemmaPlanner`` asks the LLM for one JSON tool call per step (ReAct style: it sees the record summary, the
candidate windows, what it has done and what each tool returned). ``RulePlanner`` is the deterministic policy:
review windows in rule-ranked order until the budget runs out. The Gemma planner falls back to it whenever the LLM
is unavailable or its output does not parse, so a run always completes.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from ecg_agent.core.prompts import extract_last_json_object
from ecg_agent.receiver.base import GenerateRequest

TOOLS = {
    "request_window": "Ask the perception agent to send one window for triage. args: {window_id, channel?} where "
                      "channel is compact | filtered | adapter; omit it to let the perception agent choose.",
    "request_detail": "Ask the perception agent for the beat-by-beat listing of one window (class, confidence, run "
                      "length). Cheap, no triage. args: {window_id}",
    "finish": "Stop reviewing and write the report. args: {}",
}


@dataclass
class Decision:
    tool: str
    args: dict = field(default_factory=dict)
    thought: str = ""
    source: str = "rule"  # "gemma" or "rule"


class RulePlanner:
    name = "rule"

    async def decide(self, view: dict) -> Decision:
        reviewed = set(view["reviewed"])
        todo = [w for w in view["candidates"] if w["id"] not in reviewed and w.get("readable", True)]
        if not todo or len(reviewed) >= view["budget"]:
            return Decision("finish", thought="Every candidate window in the budget has been reviewed.")
        w = todo[0]
        return Decision("request_window", {"window_id": w["id"]},
                        thought=f"{w['id']} is the highest-ranked unreviewed window "
                                f"({w['rule_tier']} by screening, {w['abnormal']} abnormal beats).")


SYSTEM = (
    "You are the reasoning agent of a two-agent ECG triage system. A separate perception agent has classified "
    "every heartbeat of a recording and split it into windows of up to 50 beats, ranked by its own screening. "
    "You cannot see the signal: you can only ask the perception agent for windows or beat listings. Decide what to "
    "ask for next, within a review budget. Reply with ONE JSON object and nothing else: "
    '{"thought": "<one sentence>", "tool": "<tool name>", "args": {...}}.'
)


def _prompt(view: dict) -> str:
    cands = "\n".join(
        f"  {w['id']}: beats {w['start']}-{w['start'] + w['n'] - 1}, {w['t_start_s']}-{w['t_end_s']} s, "
        f"{w['abnormal']} abnormal {w['classes']}, screening tier {w['rule_tier']}" for w in view["candidates"])
    history = "\n".join(f"  {h}" for h in view["history"][-8:]) or "  (nothing yet)"
    tools = "\n".join(f"  {k}: {v}" for k, v in TOOLS.items())
    return (f"Record summary: {json.dumps(view['summary'])}\n"
            f"Mode: {view['mode']}. Review budget: {view['budget']} windows, {len(view['reviewed'])} used.\n\n"
            f"Candidate windows (ranked by screening):\n{cands}\n\n"
            f"Tools:\n{tools}\n\nWhat you have done so far:\n{history}\n\n"
            "Priorities: review every window whose screening tier is urgent; then priority windows with the most "
            "abnormal beats; one routine window is enough to confirm a normal baseline. Finish when the budget is "
            "used or nothing useful remains. Do not request a window you have already reviewed.\n\n"
            'Example reply: {"thought": "w003 screens urgent and has not been reviewed yet.", '
            '"tool": "request_window", "args": {"window_id": "w003"}}\n\nYour JSON:')


class GemmaPlanner:
    name = "gemma"

    def __init__(self, receiver, fallback: RulePlanner | None = None, max_new_tokens: int = 160):
        self.receiver, self.fallback, self.max_new = receiver, fallback or RulePlanner(), max_new_tokens

    async def decide(self, view: dict) -> Decision:
        try:
            res = await self.receiver.generate(GenerateRequest(prompt=_prompt(view), system=SYSTEM,
                                                               max_new_tokens=self.max_new))
        except Exception as exc:  # noqa: BLE001 - the planner must never take a run down
            d = await self.fallback.decide(view)
            d.thought = f"[planner LLM unavailable: {type(exc).__name__}] {d.thought}"
            return d
        if res is None:
            return await self.fallback.decide(view)
        try:
            obj = extract_last_json_object(res.text)
            tool = str(obj.get("tool", ""))
            if tool not in TOOLS:
                raise ValueError(f"unknown tool {tool!r}")
            args = obj.get("args") or {}
            if not isinstance(args, dict):
                raise ValueError("args must be an object")
            return Decision(tool, args, str(obj.get("thought", ""))[:400], "gemma")
        except (ValueError, TypeError) as exc:
            d = await self.fallback.decide(view)
            d.thought = f"[planner output unusable: {exc}] {d.thought}"
            return d
