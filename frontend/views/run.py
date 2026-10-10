"""One recording: the ECG with what each agent found, the agents' conversation, findings, questions, audit trail."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import api
import ui

run_id = st.query_params.get("run") or st.session_state.get("run_id")
if not run_id:
    st.info("Open a recording from the worklist.")
    st.page_link(ui.PAGES["worklist"], label="Go to the worklist", icon="📋")
    st.stop()
st.query_params["run"] = run_id  # the URL now points at this recording: shareable, survives a refresh

try:
    d = api.run(run_id)
except api.ApiError as e:
    st.error(e.detail if e.status != 404 else "This recording is no longer stored (the demo keeps the latest 50).")
    st.stop()

running = d["status"] in ("queued", "running")
report = d.get("report") or {}

# ----- header -----
st.title(f"🫀 {d['params']['label']}")
st.caption(f"{ui.STATUS_ICON.get(d['status'], '')} {d['status']} · mode {d['params']['mode']} · run {run_id}")
c = st.columns(4)
c[0].metric("Most urgent tier", ui.tier_badge(report.get("overall_tier")) if report.get("overall_tier") else "—")
c[1].metric("Beats", (d.get("summary") or {}).get("beats", "—"))
c[2].metric("Windows reviewed", len(d["findings"]))
c[3].metric("Need a person", len(report.get("needs_human_review", [])))
if d["status"] == "unreadable":
    st.error(f"**Not triaged:** {report.get('error', 'the recording could not be read')}. A person needs to look "
             "at this recording; the system does not guess.", icon="⚠️")
for w in report.get("quality_warnings", []):
    st.warning(w, icon="📉")
if report.get("narrative"):
    src = report.get("narrative_source", "")
    who = ("a fixed template from the findings (an LLM summary was rejected by the grounding check)"
           if "rejected" in src else "a fixed template from the findings (no LLM available)"
           if src.startswith("template") else ui.source_label(src))
    st.info(f"{report['narrative']}\n\n*Summary written by {who}.*", icon="📝")


# ----- live agent conversation -----
def render_step(s: dict) -> str | None:
    d_, a = s["data"], ui.AGENT_ICON.get(s["agent"], "•")
    t = f"`{s['t_ms'] / 1000:6.1f}s`"
    if s["kind"] == "message":
        m = d_["message"]
        cc = m["content"]
        extra = " · ".join(str(x) for x in (cc.get("window_id"), cc.get("channel"), cc.get("purpose"))
                           if x and x != "review")
        why = f" — _{cc['reason']}_" if cc.get("reason") else ""
        return (f"{t} {ui.AGENT_ICON.get(m['from'], '')} → {ui.AGENT_ICON.get(m['to'], '')} **{m['intent']}** "
                f"{extra}{why}")
    if s["kind"] == "guardrail":
        bad = any(w in s["title"] for w in ("Rejected", "Refused", "refused", "failed", "cannot"))
        return f"{t} 🛡️ {'❗' if bad else '✔️'} **{s['title']}**"
    if s["kind"] == "plan":
        thought = f" — _{d_['thought']}_" if d_.get("thought") else ""
        return f"{t} {a} plans **{s['title']}** ({d_.get('planner')}){thought}"
    if s["kind"] == "tool" and "Triaged" in s["title"]:
        return f"{t} {a} {s['title']} · {ui.source_label(d_.get('source'))} · {d_.get('latency_ms') or 0:.0f} ms"
    return f"{t} {a} {s['title']}"


key = f"steps_{run_id}"
st.session_state.setdefault(key, [])


@st.fragment(run_every=1.0 if running else None)
def conversation():
    seen = st.session_state[key]
    try:
        new = api.steps(run_id, after=len(seen) - 1)
    except api.ApiError:
        return
    seen.extend(new["steps"])
    only_msgs = st.toggle("Messages and guardrails only", value=False, key=f"filter_{run_id}")
    box = st.container(height=420)
    for s in seen:
        if only_msgs and s["kind"] not in ("message", "guardrail"):
            continue
        line = render_step(s)
        if line:
            box.markdown(line)
    if running and new["done"]:
        st.rerun()  # the run just ended: redraw the whole page with the results


with st.expander("💬 The agents' conversation" + (" (live)" if running else ""), expanded=running):
    st.caption("🫀 perception agent · 🧠 reasoning agent · 🛡️ orchestrator (checks every message) · 🩺 clinician")
    conversation()
if running:
    st.stop()  # the rest needs a finished run; the fragment above reruns the page when it ends

# ----- the ECG -----
windows = {w["id"]: w for w in d["windows"]}
findings = d["findings"]
duration = (d.get("summary") or {}).get("duration_s") or 0
tab_ecg, tab_find, tab_ask, tab_audit = st.tabs(["📈 ECG", "🔎 Findings", "🩺 Ask", "🧾 Audit trail"])


def shade(fig, w0=0.0, w1=1e9):
    """Background bands: reviewed windows in their final tier's colour, unreadable windows grey."""
    for w in windows.values():
        if w["t_end_s"] < w0 or w["t_start_s"] > w1:
            continue
        f = findings.get(w["id"])
        if not w.get("readable", True):
            fig.add_vrect(x0=w["t_start_s"], x1=w["t_end_s"], fillcolor="#7f7f7f", opacity=0.25, line_width=0,
                          annotation_text=f"{w['id']} unreadable", annotation_position="top left")
        elif f:
            fig.add_vrect(x0=w["t_start_s"], x1=w["t_end_s"], fillcolor=ui.TIER_COLOR[f["final_tier"]], opacity=0.10,
                          line_width=0, annotation_text=w["id"], annotation_position="top left")


with tab_ecg:
    if not duration:
        st.info("No signal to show.")
    else:
        ov = api.signal(run_id, 0, duration, 1500)
        fig = go.Figure(go.Scattergl(x=ov["t"], y=ov["v"], mode="lines", line={"width": 0.6, "color": "#555"},
                                     hoverinfo="skip"))
        shade(fig)
        fig.update_layout(height=170, margin={"l": 10, "r": 10, "t": 25, "b": 10}, showlegend=False,
                          title={"text": "Whole recording · coloured bands = windows the reasoning agent reviewed",
                                 "font": {"size": 12}})
        st.plotly_chart(fig, use_container_width=True)

        first = min((windows[w]["t_start_s"] for w, f in findings.items() if f["final_tier"] == "urgent"
                     and w in windows), default=0.0)
        a, b = st.slider("Zoom (seconds)", 0.0, float(duration), (float(first), float(min(duration, first + 12))),
                         step=1.0)
        z = api.signal(run_id, a, b, 4000)
        fig = go.Figure(go.Scattergl(x=z["t"], y=z["v"], mode="lines", line={"width": 1, "color": "#333"},
                                     name="ECG", hoverinfo="skip"))
        beats = z["beats"]
        if beats and z["t"]:
            ys = np.interp([bt["t"] for bt in beats], z["t"], z["v"])
            for label in "NSVFQ":
                idx = [i for i, bt in enumerate(beats) if bt["label"] == label]
                if not idx:
                    continue
                fig.add_trace(go.Scatter(
                    x=[beats[i]["t"] for i in idx], y=[ys[i] for i in idx], mode="markers",
                    name=f"{label} ({ui.CLASS_NAME[label]})",
                    marker={"color": ui.CLASS_COLOR[label], "size": 9 if label != "N" else 5,
                            "symbol": "circle" if label == "N" else "diamond"},
                    customdata=[[beats[i]["confidence"], beats[i]["tier"], beats[i]["reference"] or "—",
                                 beats[i]["window"]] for i in idx],
                    hovertemplate="%{x:.2f} s · " + ui.CLASS_NAME[label] + " · confidence %{customdata[0]:.2f}"
                                  "<br>tier %{customdata[1]} · cardiologist label %{customdata[2]} · %{customdata[3]}"
                                  "<extra></extra>"))
        shade(fig, a, b)
        fig.update_xaxes(range=[a, b])  # a review band can extend past the zoom; keep the axis on the zoom
        fig.update_layout(height=380, margin={"l": 10, "r": 10, "t": 30, "b": 10}, xaxis_title="seconds",
                          legend={"orientation": "h", "y": -0.2})
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Markers are the perception agent's beat labels; hover for its confidence and, on MIT-BIH "
                   "records, the cardiologist's annotation.")

# ----- findings -----
with tab_find:
    if not findings:
        st.info("No window was reviewed.")
    else:
        def attempts_text(f):
            sym = {"agree": "✓", "under_triage": "✗ too low", "over_triage": "↑ more cautious", "unparsed": "✗ no answer",
                   "refused": "✗ refused"}
            return " → ".join(f"{t['channel']}: {t['tier'] or '—'} {sym.get(t['verdict'], t['verdict'])}"
                              for t in f["attempts"])

        rows = [{"window": w, "time": f"{windows[w]['t_start_s']:.0f}-{windows[w]['t_end_s']:.0f} s" if w in windows
                 else "", "final tier": ui.tier_badge(f["final_tier"]), "screening": f["screening_tier"],
                 "resolution": f["resolution"].replace("_", " ") + (" · asked by a question"
                                                                     if f.get("origin") == "question" else ""),
                 "receiver attempts": attempts_text(f), "answered by": ui.source_label(f["attempts"][-1]["source"]),
                 "override": f"{f['override']['to_tier']} by {f['override']['clinician']}" if f.get("override") else ""}
                for w, f in sorted(findings.items(), key=lambda kv: (-["routine", "priority", "urgent"]
                                                                    .index(kv[1]["final_tier"]), kv[0]))]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption("Each row is one window the reasoning agent reviewed. A ✗ is an answer the guardrail rejected "
                   "(below the screening tier, or unreadable); the window was then re-sent over another channel.")
        wid = st.selectbox("Window", sorted(findings))
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Explain this finding", type="primary"):
                with st.spinner("Writing the explanation..."):
                    try:
                        st.session_state[f"exp_{run_id}_{wid}"] = api.explain(run_id, wid)
                    except api.ApiError as e:
                        st.error(e.detail)
            e = st.session_state.get(f"exp_{run_id}_{wid}") or findings[wid].get("explanation")
            if e:
                st.markdown(f"> {e['justification']}\n>\n> *Guideline:* {e['guideline_fact']}")
                st.caption(f"Explained by {ui.source_label(e['explained_by'])}"
                           + (f" · {e['note']}" if e.get("note") else ""))
        with c2:
            f = findings[wid]
            current = f["override"]["to_tier"] if f.get("override") else f["final_tier"]
            with st.form(f"ovr_{wid}", clear_on_submit=True):
                st.markdown(f"**Clinician override** · {wid} is {ui.tier_badge(current)}")
                if f.get("override"):
                    o = f["override"]
                    st.caption(f"Overridden {o['from_tier']} → {o['to_tier']} by {o['clinician']}: {o['reason']}")
                # Only the other tiers: an override must change something. Lowering a tier is allowed (it is the
                # clinician's call, unlike the agents, which can never go below screening); it is recorded either way.
                tier = st.selectbox("New tier", [t for t in ("urgent", "priority", "routine") if t != current])
                who = st.text_input("Your name")
                why = st.text_input("Reason", placeholder="e.g. artefact on review of the strip")
                if st.form_submit_button("Override"):
                    if not who.strip() or not why.strip():
                        st.error("An override needs your name and a reason.")
                    else:
                        try:
                            api.override(run_id, wid, tier, who, why)
                            # a toast survives the rerun; st.success would vanish with it
                            st.toast(f"{wid}: {current} → {tier}, recorded in the audit trail.", icon="🩺")
                            st.rerun()
                        except api.ApiError as e:
                            st.error(e.detail)

# ----- questions -----
with tab_ask:
    st.caption("Ask about this recording: why a window got its tier, which beats are behind it, whether another part "
               "looks abnormal, how the channels did. Medical advice and changing tiers are refused by design.")
    for a in d.get("answers", []):
        with st.chat_message("user", avatar="🩺"):
            st.markdown(a["question"])
        with st.chat_message("assistant", avatar="🧠" if a["type"] != "refused" else "🛡️"):
            st.markdown(a["text"])
            meta = [a["type"]] + ([f"cites {', '.join(a['citations'])}"] if a.get("citations") else [])
            if a.get("data", {}).get("refusal"):
                meta.append(f"refused by {a['data'].get('layer', 'safety patterns')}")
            st.caption(" · ".join(meta))
    q = st.chat_input("Ask a question about this recording")
    if q:
        with st.spinner("The reasoning agent is working (it may consult the perception agent)..."):
            try:
                api.ask(run_id, q)
            except api.ApiError as e:
                st.error(e.detail)
        st.rerun()

# ----- audit -----
with tab_audit:
    rows = api.audit(run_id)
    decisions = [r for r in rows if r.get("verdict") == "clinician_override"]
    if decisions:  # human decisions first: they are what a reviewer of this run looks for
        st.markdown("**Clinician decisions**")
        for r in decisions:
            st.markdown(f"- {r['window_id']}: {ui.tier_badge(r['from_tier'])} → {ui.tier_badge(r['to_tier'])} "
                        f"by **{r['clinician']}**: {r['reason']}")
    st.caption("Every message the orchestrator checked, with its verdict, and every clinician override, in order.")
    st.dataframe(pd.DataFrame([{
        "verdict": r.get("verdict"), "from": r.get("from", r.get("clinician", "")), "to": r.get("to", ""),
        "intent": r.get("intent", "override"), "window": (r.get("content") or {}).get("window_id", r.get("window_id")),
        "channel": (r.get("content") or {}).get("channel", ""), "reason / detail": r.get("reason") or "",
        "vectors sha256": (r.get("vectors_sha256") or "")[:12]} for r in rows]),
        hide_index=True, use_container_width=True)
