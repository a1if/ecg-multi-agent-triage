# ECG multi-agent triage

Two cooperating agents triage long ECG recordings for clinicians: a **perception agent** (a CNN-LSTM on CPU) reads
every heartbeat and checks the signal it is given; a **reasoning agent** (Gemma 4 E4B, self-hosted) decides which parts
deserve attention, triages them, and answers the clinician's questions. They talk through typed messages, over text or
learned **virtual tokens**, and a deterministic orchestrator enforces what each may say. A guardrail never lets an
LLM answer fall below the classifier's screening tier, and a clinician signs off.

> **Status: work in progress.** Agents, services and evaluation are built and tested; frontend, containers, CI/CD and
> cloud deployment are next. Research prototype, not a medical device or clinical advice.

Built on the research in
[Heterogeneous-Multi-Agent-Edge-AI-for-Clinical-Decision-Support](https://github.com/a1if/Heterogeneous-Multi-Agent-Edge-AI-for-Clinical-Decision-Support)
(the latent channel between a non-transformer sender and a frozen LLM).

## What is here

| | |
|---|---|
| `backend/src/ecg_agent/agent/` | The two agents (LangGraph), the orchestrator, message policy, guardrails, grounding checks, clinician questions |
| `backend/src/ecg_agent/inference/` | GPU service: Gemma + adapter, constrained decoding, decide early / explain later |
| `backend/src/ecg_agent/api/` | API service: runs, worklist, live agent trace (SSE), signal, questions, overrides |
| `backend/tests/`, `backend/scripts/eval_system.py` | 74 tests; system evaluation with injected faults, signal faults and real Gemma |
| `docs/design.md` | Design: use case, stakeholders, autonomy, orchestrator, guardrails, protocol, hazard log |
| `docs/benchmarks.md` | Measurements: inference baseline and optimisation, signal quality, system evaluation |
| `docs/TESTING.md` | How to run and test everything |

## Highlights so far

- **Safety holds under faults:** 63 runs with injected LLM errors, 343/343 caught; no finding ever below screening.
- **Fails safely on bad signals:** noise, clipping and electrode-off recordings end as *unreadable*, never *routine*
  (70/70 fault scenarios).
- **9× faster reviews** from deciding at the tier token and explaining only on demand (tier identical 45/45).
- **Graceful degradation:** GPU killed mid-run, the run completes with every answer's source labelled.

## Quick start

See [docs/TESTING.md](docs/TESTING.md). In short, from `backend/`:

```bash
python -m venv .venv && .venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python -m pip install -e ".[api,dev]"
.venv/Scripts/python -m pytest -q
.venv/Scripts/python scripts/serve_local.py        # then open http://127.0.0.1:8000/docs
```

## Licence and data

Code: MIT. Bundled ECG records: MIT-BIH Arrhythmia Database, ODC-By (see `backend/data/records/README.md`).
Gemma is used under the [Gemma licence](https://ai.google.dev/gemma/terms); its weights are not included.
