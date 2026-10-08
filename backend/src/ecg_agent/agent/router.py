"""Channel choice, from the paper's measurements.

* Filtered text is the most accurate channel (0.11-0.17 balanced accuracy above the adapter), because the question
  is known in advance, but its prompt grows with the number of abnormal beats.
* The adapter costs 4 tokens per beat whatever the beats are, so its cost is flat and predictable; against compact
  text it cuts prompt processing by 24/50/78% at 10/20/50 events.
* Compact text (every beat) is dominated by both and is used only when asked for explicitly.

Token estimates are fitted to Gemma 4 E4B's tokenizer on the paper's prompts (see ``scripts/fit_token_model.py``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Mode = Literal["accuracy", "balanced", "throughput"]

# prompt tokens ~ base + per_event * n (+ per_listed * abnormal for filtered); refit with scripts/fit_token_model.py
TOKEN_MODEL = {
    "compact": {"base": 525.0, "per_event": 58.0, "per_listed": 0.0},
    "filtered": {"base": 590.0, "per_event": 0.0, "per_listed": 63.0},
    "adapter": {"base": 513.0, "per_event": 4.0, "per_listed": 0.0},
}
# Fallback order when the guardrail rejects an answer: move to the most accurate channel not yet tried.
ESCALATION = ("filtered", "compact", "adapter")


def estimate_tokens(channel: str, n: int, abnormal: int) -> int:
    m = TOKEN_MODEL[channel]
    return int(m["base"] + m["per_event"] * n + m["per_listed"] * abnormal)


@dataclass(frozen=True)
class Route:
    channel: str
    reason: str
    est_tokens: dict[str, int]


def choose_channel(n: int, abnormal: int, mode: Mode = "balanced", adapter_max: int = 50) -> Route:
    est = {c: estimate_tokens(c, n, abnormal) for c in TOKEN_MODEL}
    if n > adapter_max:
        return Route("filtered", f"{n} beats exceed the adapter's {adapter_max}-event limit", est)
    if mode == "accuracy":
        return Route("filtered", "accuracy mode: filtered text is the most accurate channel", est)
    if mode == "throughput":
        return Route("adapter", f"throughput mode: adapter costs a flat {est['adapter']} tokens", est)
    if est["filtered"] <= 1.5 * est["adapter"]:
        return Route("filtered", f"{abnormal} abnormal beats keep filtered text small "
                                 f"({est['filtered']} vs {est['adapter']} tokens), so take the more accurate channel",
                     est)
    return Route("adapter", f"{abnormal} abnormal beats would make filtered text {est['filtered']} tokens; "
                            f"the adapter carries all {n} in {est['adapter']}", est)


def next_channel(tried: list[str]) -> str | None:
    return next((c for c in ESCALATION if c not in tried), None)
