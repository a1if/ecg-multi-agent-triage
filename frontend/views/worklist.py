"""Worklist: every recording, the one needing attention first. Start new runs here."""
import time

import pandas as pd
import streamlit as st

import api
import ui

st.title("📋 Worklist")
st.caption("Recordings ranked for review: unreadable first (they need a person), then urgent, then the rest. "
           + ui.DISCLAIMER)
ui.status_banner(st.session_state.get("status"))


def open_run(run_id: str) -> None:
    st.session_state["run_id"] = run_id
    st.switch_page(ui.PAGES["run"])


# ----- new run -----
with st.expander("➕ Analyse a recording", expanded=not st.session_state.get("has_runs", False)):
    try:
        recs, scen = api.records(), api.scenarios()
    except api.ApiError as e:
        st.error(e.detail)
        st.stop()
    status = st.session_state.get("status") or {}
    presets = bool(status.get("demo_presets"))
    if presets:
        st.caption("Public demo: each record's first 5 minutes with a 6-window review budget, the settings whose "
                   "answers were recorded from real Gemma runs on a GPU. Answers say where they came from.")
    with st.form("new_run"):
        c1, c2, c3 = st.columns(3)
        rec = c1.selectbox("Record (MIT-BIH, held-out patients)", [r["id"] for r in recs],
                           index=[r["id"] for r in recs].index("233") if any(r["id"] == "233" for r in recs) else 0)
        full = next(r["duration_s"] for r in recs if r["id"] == rec)
        max_d = float(status.get("max_duration_s", 600))
        start, dur = 0.0, 300.0
        if not presets:
            start = c2.number_input("Start (s)", 0.0, max(0.0, full - 30), 0.0, step=30.0)
            dur = c3.number_input("Length (s)", 30.0, max_d, min(300.0, max_d), step=30.0)
        c4, c5, c6 = st.columns(3)
        mode = c4.selectbox("Mode", ["balanced", "accuracy", "throughput"],
                            help="balanced: the perception agent picks the channel per window; accuracy: filtered "
                                 "text (most accurate); throughput: virtual tokens (flat, low cost)")
        scenario = c5.selectbox("Stress scenario", ["none"] + scen,
                                help="Damage the signal on purpose and watch the perception agent notice")
        reviews = 6 if presets else c6.slider("Review budget (windows)", 1, 12, 6)
        go = st.form_submit_button("Start", type="primary")
    if go:
        try:
            r = api.create_run(rec, start, dur, mode, None if scenario == "none" else scenario, reviews)
        except api.ApiError as e:
            st.error(e.detail)
        else:
            open_run(r["id"])

    if status.get("uploads"):
        with st.form("upload"):
            f = st.file_uploader("Or upload a CSV (numbers only, one column per lead)", type=["csv", "txt"])
            fs = st.number_input("Sampling rate (Hz)", 50.0, 2000.0, 360.0)
            if st.form_submit_button("Upload and analyse") and f:
                try:
                    open_run(api.upload_run(f.name, f.getvalue(), fs, "balanced")["id"])
                except api.ApiError as e:
                    st.error(e.detail)
    else:
        st.caption("Uploads are off in this deployment (public demo). Run it locally with `--uploads` to analyse "
                   "your own files.")

# ----- the list -----
try:
    rows = api.worklist()
except api.ApiError as e:
    st.error(e.detail)
    st.stop()
st.session_state["has_runs"] = bool(rows)
if not rows:
    st.info("No recordings analysed yet. Start one above.")
    st.stop()

df = pd.DataFrame([{
    "status": f"{ui.STATUS_ICON.get(r['status'], '')} {r['status']}",
    "tier": ui.tier_badge(r["overall_tier"]) if r["overall_tier"] else ("—" if r["status"] in ("running", "queued")
                                                                         else "⚠️ none: needs a person"),
    "recording": r["source"],
    "needs review": len(r["needs_human_review"]),
    "warnings": "; ".join(r["quality_warnings"])[:80] or (r["error"] or "")[:80],
    "overrides": r["overrides"],
    "started": time.strftime("%H:%M:%S", time.localtime(r["created"])),
    "id": r["id"],
} for r in rows])
event = st.dataframe(df, hide_index=True, use_container_width=True, on_select="rerun", selection_mode="single-row",
                     column_config={"id": None})
if event.selection.rows:
    open_run(df.iloc[event.selection.rows[0]]["id"])
st.caption("Click a row to open the recording.")
if any(r["status"] in ("running", "queued") for r in rows):
    time.sleep(1.5)
    st.rerun()
