"""How it works: the architecture and how to read the screens."""
import streamlit as st

import ui

st.title("🧭 How it works")
st.caption(ui.DISCLAIMER)

st.markdown("""
**The job.** A cardiac monitoring service reviews long ECG recordings every morning. This system pre-reviews each one
and ranks the worklist, so the recording with a dangerous run of ventricular beats is seen first, with its evidence.
A clinician still decides.

**Two agents.**
- 🫀 **Perception agent** (CNN-LSTM, CPU): reads every beat, checks the signal is trustworthy (noise, flat
  stretches, clipping; retries with other strategies), screens the recording into windows, and answers the reasoning
  agent's requests, choosing *how* to send each window: every beat as text, only abnormal beats as text, or compact
  learned vectors ("virtual tokens") read directly by the language model.
- 🧠 **Reasoning agent** (Gemma 4 E4B): decides *what* to review within a budget, triages each window, explains
  findings and answers questions. It never sees the signal; it can only ask.

**The orchestrator** 🛡️ is plain code, not an AI: it checks every message against what each agent may say, refuses
anything else, enforces budgets and keeps the audit trail.

**Guardrails.** An answer below the classifier's screening tier is rejected and the window is re-sent over another
channel; if every channel fails, the screening tier stands and a person is asked. Unreadable recordings end as
*unreadable*, never *routine*. Explanations and summaries may only use facts in the evidence. Questions asking for
medical advice or to change a tier are refused before any language model sees them.
""")

st.graphviz_chart("""
digraph {
  rankdir=LR; node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11];
  ecg [label="ECG recording", fillcolor="#eeeeee"];
  p [label="🫀 Perception agent\\nCNN-LSTM · quality checks\\nchooses the channel", fillcolor="#e3f2fd"];
  o [label="🛡️ Orchestrator\\npolicy · budgets · audit", fillcolor="#fff3e0"];
  r [label="🧠 Reasoning agent\\nplan · triage · verify\\nexplain · answer", fillcolor="#ede7f6"];
  g [label="Gemma 4 E4B\\n(GPU service)", fillcolor="#f3e5f5"];
  c [label="🩺 Clinician\\nworklist · questions\\noverride · sign-off", fillcolor="#e8f5e9"];
  ecg -> p; p -> o [label="windows: text or\\nvirtual tokens"]; o -> r; r -> o [label="requests"]; o -> p;
  r -> g [label="triage · explain", style=dashed]; r -> c [label="findings · answers"]; c -> r [label="questions"];
}
""")

st.markdown("""
**Reading the screens.**
- *Worklist*: one row per recording, most in need of attention first.
- *Recording → conversation*: every step and every message between the agents, live while a run is going.
- *ECG*: coloured bands are windows the reasoning agent reviewed (red urgent, amber priority, green routine), grey
  bands are windows too noisy to classify; markers are beat labels (red diamonds ventricular, blue supraventricular).
- *Findings*: each window's receiver attempts. ✗ means the guardrail rejected an answer.
- *Every answer says who produced it*: Gemma live, a recorded Gemma answer, or the rule when the GPU is unavailable.

Built on the research in
[Heterogeneous-Multi-Agent-Edge-AI-for-Clinical-Decision-Support](https://github.com/a1if/Heterogeneous-Multi-Agent-Edge-AI-for-Clinical-Decision-Support).
""")
