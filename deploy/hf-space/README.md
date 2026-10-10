---
title: ECG Multi-Agent Triage
emoji: 🫀
colorFrom: red
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Two-agent ECG triage, CNN-LSTM perception + Gemma
---

# ECG multi-agent triage (demo)

Two cooperating agents triage ECG recordings for a clinician's worklist: a **perception agent** (a CNN-LSTM) reads
every heartbeat and checks the signal; a **reasoning agent** (Gemma 4 E4B) decides what to review, triages it over
text or learned virtual tokens, explains its findings and answers questions. A deterministic orchestrator enforces
what each agent may say, and a guardrail never lets an answer fall below the classifier's screening tier.

**This demo runs on free CPU hardware.** The reasoning agent's answers are **real Gemma answers recorded on a GPU**
for these presets (each bundled record, first 5 minutes), labelled *Gemma (recorded answer)*. Anything outside the
recordings falls back to the triage rule and is labelled as such. Try a stress scenario (noise, flat stretches) and
a question; medical-advice questions are refused by design.

Research prototype, **not a medical device and not clinical advice.** ECG data: MIT-BIH Arrhythmia Database
(PhysioNet, ODC-By), de-identified. Gemma is used under the Gemma terms.

Source, design notes and evaluation: https://github.com/a1if/ecg-multi-agent-triage
