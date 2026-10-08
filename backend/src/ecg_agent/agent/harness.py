"""The harness around both agents: tracing, budgets and tool calls with timeouts.

Every node and tool call is traced as a step (streamed to the UI and kept in the run record). Budgets bound each
loop, so a confused planner or a dead GPU can slow a run down but never hang it.
"""
from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from ecg_agent.observability import metrics


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Budget:
    max_turns: int = 40  # agent turns in the conversation
    max_reviews: int = 6  # windows the receiver LLM reviews
    max_attempts: int = 3  # channels tried per window before the guardrail decides
    max_plan_steps: int = 4  # planner retries inside one reasoning turn
    max_questions: int = 10  # clinician questions per run
    max_question_reviews: int = 3  # extra windows questions may have reviewed, per run
    receiver_timeout_s: float = 300.0  # covers a Cloud Run GPU cold start


@dataclass
class Tracer:
    """Ordered record of every step of a run. Listeners (e.g. live browser streams) are told when a step is added;
    they read ``steps`` themselves, so a slow listener never holds up the agents."""
    run_id: str
    steps: list[dict] = field(default_factory=list)
    listeners: set[asyncio.Event] = field(default_factory=set)
    t0: float = field(default_factory=time.perf_counter)

    async def emit(self, agent: str, kind: str, title: str, **data: Any) -> None:
        step = {"seq": len(self.steps), "t_ms": round((time.perf_counter() - self.t0) * 1e3, 1), "agent": agent,
                "kind": kind, "title": title, "data": data}
        self.steps.append(step)
        for ev in self.listeners:
            ev.set()


async def call_tool(tracer: Tracer, agent: str, tool: str, fn: Callable[[], Awaitable[Any]],
                    timeout_s: float | None = None, **trace_args: Any) -> Any:
    """Run one tool call under a timeout, tracing its start, duration and failure."""
    t = time.perf_counter()
    try:
        result = await asyncio.wait_for(fn(), timeout_s) if timeout_s else await fn()
    except Exception as exc:
        metrics.tool_calls.labels(agent=agent, tool=tool, status="error").inc()
        await tracer.emit(agent, "tool_error", f"{tool} failed: {type(exc).__name__}", error=str(exc)[:300],
                          **trace_args)
        raise
    ms = (time.perf_counter() - t) * 1e3
    metrics.tool_calls.labels(agent=agent, tool=tool, status="ok").inc()
    metrics.tool_latency.labels(agent=agent, tool=tool).observe(ms / 1e3)
    return result
