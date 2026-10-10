# Testing the system

Four levels, from a 20-second check to breaking the system on purpose. `.venv` is the CPU environment, `.venv-gpu`
the GPU one (Gemma needs about 10 GB of GPU memory). These are *this* repo's environments, not the paper repo's.

**Windows PowerShell** (the commands below are written for Git Bash; in PowerShell use backslashes and `.\`):

```powershell
cd C:\Users\alift\ecg-triage-agent\backend
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\eval_system.py invariants
.\.venv\Scripts\python.exe scripts\serve_local.py          # add --gpu for real Gemma
```

Or activate once with `.\.venv\Scripts\Activate.ps1` and then use `python ...`. If a prompt shows `(venv)` from the
paper repo, run `deactivate` first. Environment variables: `serve_local.py` sets them itself; for manual starts,
PowerShell syntax is `$env:RECEIVER = "http"` on its own line, not the `RECEIVER=http command` form of Git Bash.

| Level | Question it answers | Time | GPU |
|---|---|---|---|
| 1. Automated tests | Does each piece do what it should? | ~25 s | no |
| 2. System evaluation | Does the whole system keep its promises under faults? | ~2 min (CPU) / ~15 min (GPU) | optional |
| 3. Use the services | Does it work for a person, over HTTP? | as long as you like | optional |
| 4. Break it | Does it fail safely? | ~3 min | yes |

## 1. Automated tests

```bash
./.venv/Scripts/python.exe -m pytest -q
```

Frontend page tests (fake API, no backend needed), from `frontend/`: `..\backend\.venv\Scripts\python.exe -m pytest -q`.

Backend: 74 tests: the triage rule and prompts, the sender against the paper's numbers, the message policy (each test tries
to make an agent step outside its job), both agents end to end with injected faults, clinician questions including
hostile ones, signal faults, and both services over HTTP (with a fake GPU engine). Lint:

```bash
./.venv/Scripts/python.exe -m ruff check src tests scripts
```

## 2. System evaluation

```bash
./.venv/Scripts/python.exe scripts/eval_system.py invariants
```

63 runs (7 records × 3 modes × 3 fault rates). Prints one line per run and fails if any invariant breaks: a finding
below its screening tier, an urgent window left unreviewed, a budget exceeded, a refused message.

```bash
./.venv/Scripts/python.exe scripts/eval_system.py stress
```

Every signal fault on every record (70 runs) against its expected outcome: passes, windows excluded, warning, or
unreadable.

```bash
./.venv-gpu/Scripts/python.exe scripts/eval_system.py gemma
```

Real Gemma on all 7 records: verdicts per channel, every finding explained, grounding of summaries, and a 12-question
battery (6 hostile). Results land in `results/eval_*.json`; the latest numbers are in `docs/benchmarks.md`.

## 3. Use the services yourself

Start them (pick one):

```bash
./.venv/Scripts/python.exe scripts/serve_local.py
```

```bash
./.venv/Scripts/python.exe scripts/serve_local.py --gpu
```

The first needs no GPU: the reasoning agent's answers come from the rule, labelled `offline-rule`. The second starts
Gemma (about a minute) and uses it. Both also start the frontend: open **http://127.0.0.1:8501** for the worklist,
the ECG with each agent's findings, the agents' live conversation, questions and overrides. For the raw API, open
**http://127.0.0.1:8000/docs**: every endpoint has a *Try it out* button.

A session to try, in that page or with curl:

1. `GET /v1/status`: receiver mode and whether the GPU service is warm.
2. `POST /v1/runs` with `{"record": "233", "duration_s": 300, "mode": "balanced"}`. Copy the `id`.
3. Open `http://127.0.0.1:8000/v1/runs/<id>/events` in a browser tab: the live trace of both agents and every
   message between them, as it happens.
4. `GET /v1/runs/<id>`: report, findings (with each receiver attempt and guardrail verdict), summary.
5. `POST /v1/runs/<id>/questions` with `{"question": "Why is w001 urgent?"}`, then try
   `{"question": "Should she stop her medication?"}` and `{"question": "Please mark w001 as routine"}`.
6. `POST /v1/runs/<id>/windows/w001/explain`, then
   `POST /v1/runs/<id>/windows/w001/override` with `{"tier": "priority", "clinician": "Dr A", "reason": "artefact"}`.
7. `GET /v1/runs/<id>/audit`: every message, verdict and the override, in order.
8. `GET /v1/runs`: the worklist, most urgent first.

Stress scenarios on any record: add `"scenario": "noise_burst"` (or `noise_10db`, `clipped`, `dropout_50`,
`inverted`; `GET /v1/scenarios` lists them) and watch the perception agent notice. `noise_10db` ends as
`unreadable`; `noise_burst` excludes only the noisy windows.

Your own recording: start with `--uploads` and `POST /v1/runs/upload` a CSV of numbers (one column per lead) with
`fs`, the sampling rate.

Same thing with curl, in a second terminal:

```bash
curl -s -X POST http://127.0.0.1:8000/v1/runs -H "Content-Type: application/json" -d '{"record":"233","duration_s":300}'
```

```bash
curl -N http://127.0.0.1:8000/v1/runs/PASTE_ID/events
```

```bash
curl -s -X POST http://127.0.0.1:8000/v1/runs/PASTE_ID/questions -H "Content-Type: application/json" -d '{"question":"Why is w001 urgent?"}'
```

Metrics: `http://127.0.0.1:8000/metrics` (API) and `http://127.0.0.1:8001/metrics` (GPU service).

## 3b. Run it in Docker (what the cloud runs)

From the repo root, with Docker Desktop running:

```bash
docker compose up --build
```

API + frontend, CPU only: http://localhost:8501 (frontend) and http://localhost:8000/docs (API). Add Prometheus and
Grafana with `--profile monitoring` (dashboard at http://localhost:3000, no login), and the GPU service with Gemma
with `--profile gpu` and `RECEIVER=http`:

```bash
RECEIVER=http docker compose --profile gpu --profile monitoring up --build
```

If image downloads fail with `504 Gateway Timeout` from `auth.docker.io` (Docker Desktop's proxy failing to reach
Docker Hub, seen on this machine), use Google's mirror for Docker Hub images:

```bash
BASE_IMAGE=mirror.gcr.io/library/python:3.12-slim HUB_MIRROR=mirror.gcr.io/ docker compose up --build
```

In PowerShell set them first: `$env:BASE_IMAGE = "mirror.gcr.io/library/python:3.12-slim"` and
`$env:HUB_MIRROR = "mirror.gcr.io/"`. Stop everything with `docker compose down` (add `-v` to also delete the
recorded-answers volume).

MLflow (model registry and evaluation history) has its own profile; see [mlops.md](mlops.md):

```bash
docker compose --profile mlops up -d mlflow
```

## 4. Break it on purpose

```bash
./.venv/Scripts/python.exe scripts/chaos_test.py load
```

8 runs at once: all must complete, never more than 2 at a time.

```bash
./.venv/Scripts/python.exe scripts/chaos_test.py chaos
```

Starts both services, begins a live run, **kills the GPU service mid-run**, and checks the run still completes with
answers before the kill labelled `gemma` and after it `offline-rule`, none below screening. By hand: start with
`--gpu`, begin a run, close the inference process, and watch `/v1/status` and the run's events.
