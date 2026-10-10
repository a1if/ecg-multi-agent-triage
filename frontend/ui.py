"""Shared look: one colour per tier and per beat class everywhere, and a visible source on every answer."""
from __future__ import annotations

import streamlit as st

TIER_COLOR = {"urgent": "#d62728", "priority": "#ff9f1c", "routine": "#2ca02c", None: "#7f7f7f"}
TIER_ICON = {"urgent": "🔴", "priority": "🟠", "routine": "🟢"}
STATUS_ICON = {"complete": "✅", "running": "⏳", "queued": "⏳", "unreadable": "⚠️", "failed": "❌",
               "incomplete": "❔"}
CLASS_COLOR = {"N": "#9e9e9e", "S": "#1f77b4", "V": "#d62728", "F": "#ff7f0e", "Q": "#000000"}
CLASS_NAME = {"N": "normal", "S": "supraventricular", "V": "ventricular", "F": "fusion", "Q": "unclassifiable"}
AGENT_ICON = {"perception": "🫀", "reasoning": "🧠", "orchestrator": "🛡️", "clinician": "🩺"}
SOURCE_LABEL = {"gemma": "Gemma (live)", "replay": "Gemma (recorded answer)", "offline-rule": "rule (offline)",
                "mock": "simulated", "rule": "rule"}

DISCLAIMER = "Research prototype, not a medical device and not clinical advice."
PAGES: dict = {}  # set by app.py: name -> st.Page, so pages can link to each other wherever the entry point is


def tier_badge(tier: str | None) -> str:
    return f"{TIER_ICON.get(tier, '⚪')} {tier or 'no tier'}"


def source_label(source: str | None) -> str:
    if not source:
        return "?"
    for key, label in SOURCE_LABEL.items():
        if source.startswith(key):
            return label + source[len(key):]
    return source


def status_banner(status: dict | None) -> None:
    """Which receiver answers right now: users must never mistake the rule or a recording for live Gemma."""
    if status is None:
        st.error("The API is not reachable. Start it with `python scripts/serve_local.py` in `backend/`.")
        return
    mode, health = status["receiver"], status["receiver_health"]
    if mode == "http":
        state = health.get("live", {}).get("state", "unknown")
        msg = {"ready": "Gemma is live on the GPU service.",
               "loading": "The GPU service is starting (loading Gemma); answers come from recordings or the rule.",
               "unreachable": "The GPU service is asleep or down: answers come from recorded Gemma answers where "
                              "available, otherwise from the rule, and are labelled."}.get(state, state)
        (st.success if state == "ready" else st.warning)(f"**Receiver:** {msg}", icon="🧠")
    elif mode == "replay":
        st.info("**Receiver:** recorded Gemma answers (the GPU service is not used in this deployment).", icon="🧠")
    else:
        st.info("**Receiver:** offline mode. The reasoning agent's answers come from the triage rule, labelled "
                "`rule (offline)`; start the GPU service for Gemma.", icon="🧠")
