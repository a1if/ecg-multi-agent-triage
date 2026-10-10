# MLOps: model registry and quality alerts

Two questions a deployed model has to answer at any time:

1. **Which model is running, where did it come from, and why that one?** → MLflow tracking and model registry.
2. **Is it still behaving the way it did when we measured it?** → Prometheus metrics and alert rules for model
   quality and input drift.

## 1. Model registry (MLflow)

```bash
docker compose --profile mlops up -d mlflow          # tracking server + registry UI: http://localhost:5000
cd backend && pip install -e ".[mlops]"
python scripts/mlops_registry.py --paper-repo PATH/TO/PAPER  # build (idempotent: re-running changes nothing)
python scripts/mlops_registry.py --check                      # exit 1 if deployment is not the registry champion
```

`--paper-repo` is optional. With it, the script also reads the paper repo's held-out results and registers the other
adapter seeds trained there.

What the script records:

| Where | What |
|---|---|
| experiment `training` | One run per model file, holding its parameters, training or held-out metrics, SHA-256, the checkpoint file itself, and for the adapter the SHA-256 of the sender it was trained against |
| registry `ecg-sender` | The perception agent's CNN-LSTM (v1, champion) |
| registry `ecg-adapter` | The virtual-token adapter: seed 303 (champion, promoted by the gate below), seeds 101 and 202 (challengers; 101 was deployed first) |
| experiment `evaluation` | One run per result file in `backend/results/` (system evaluation, stress, real-Gemma, inference benchmarks, chaos), each tagged with the model versions and SHA-256s it tested |

**How it ties to deployment.** The services don't load from MLflow. They load the files in
`backend/artifacts/models/`, pinned by SHA-256 in `manifest.json`, and the orchestrator refuses virtual tokens from a
sender whose SHA-256 differs from the one the adapter was trained for. The registry is the record around those files.
The `champion` alias is set to the version whose SHA-256 is in the manifest, and `--check` fails if the two ever
disagree. It can run as a gate before a deploy.

Why not serve straight from the registry? A demo with no paid infrastructure can't depend on a tracking server being
up, and pinning by content hash means every image is reproducible on its own. In a team setting the next step would be
a CI job that downloads `models:/ecg-adapter@champion`, checks its SHA-256 against the manifest and builds the image.

### Promoting a model: the gate

The first deployed adapter was seed 101. On the training run's validation split it scored lowest of the three seeds
(0.72 balanced accuracy vs 0.80 and 0.82), but that split is about 30 windows per size, too small to decide on. The
question for a promotion is whether a challenger is better **on patients it has never seen**, by more than noise, and
not worse where it matters most.

`backend/scripts/promotion_gate.py` answers it with rules fixed before any interval was computed:

| Rule | Why |
|---|---|
| Held-out patients only (MIT-BIH DS2: 795 windows, 22 patients), every seed on the same windows | Paired: which windows were drawn cannot favour a seed |
| Bootstrap over **patients**, not windows (10,000 resamples) | Windows from one patient are correlated; per-window resampling makes intervals look tighter than they are |
| 97.5% intervals | Two challengers against one champion (Bonferroni) |
| Promote only if balanced accuracy is better (interval above 0) **and** urgent recall is not worse by more than 0.05 **and** answers parse as often | A model can win on average while getting worse at the one class that matters most |

Result (`backend/results/promotion_gate.json`):

| Adapter | Balanced accuracy | Urgent recall | vs seed 101 |
|---|---|---|---|
| seed 101 (was deployed) | 0.800 | 0.731 | |
| seed 202 | 0.821 | 0.648 | **fails**: gain within noise [−0.018, +0.058], and urgent recall −0.083 [−0.170, +0.007] |
| seed 303 | 0.850 | 0.742 | **passes**: +0.050 [+0.020, +0.079], urgent recall +0.011 [−0.042, +0.082] |

Seed 202 is the instructive one: picked by average accuracy it would have looked like an upgrade, while missing about
8 more urgent windows in every 100.

**What the promotion changed, together, in one PR:** the adapter file and its SHA-256 in `manifest.json` (every loader
reads the adapter from the manifest, so nothing else names a file); the demo's recorded answers, re-recorded with real
Gemma through the new adapter (282 answers; `record_demo_answers.py --verify` then found 0 misses both in the API
image and in a fresh install like Streamlit Community Cloud's); the real-Gemma system evaluation and the inference benchmark, re-run (same safety and quality results; see
[benchmarks.md](benchmarks.md)); the registry's
`champion` alias. Recorded answers on the adapter channel are keyed by the adapter's SHA-256, so an old recording can
never be served for a new adapter. CI runs `promotion_gate.py --check`: the deployed adapter must be the gate's choice.

## 2. Quality and drift alerts (Prometheus)

`ops/alerts.yml` defines seven rules in two groups. Each threshold comes from a measured baseline
([benchmarks.md](benchmarks.md)), and each rule needs a minimum volume of events, so a handful of runs cannot fire it.

| Alert | Fires when | Baseline |
|---|---|---|
| `LLMUnderTriageRateHigh` | > 20% of receiver answers below the screening tier over 1 h (≥ 20 answers) | 8.7% (4 of 46, real Gemma) |
| `LLMUnparsedAnswersHigh` | > 5% of answers fail to parse | 0: constrained decoding |
| `ReceiverFallingBackToRule` | > 50% of answers come from the rule, **only where the GPU is meant to answer** | 0 |
| `RunsFailing` | any run fails | 0 |
| `UnreadableRecordingsSpike` | > 20% of recordings unreadable over 1 h (≥ 5 runs) | 0 of 7 clean records |
| `InputNoiseDrift` | median input noise > 0.03 over 6 h (≥ 10 recordings) | clean records 0.0004–0.016 |
| `UnreadableWindowShareHigh` | > 10% of windows too noisy to classify over 6 h | ≈ 0 |

Two of these rules need explaining:

- **The guardrail catches under-triage, so why alert on it?** No patient is put at risk, because the answer is
  raised to the screening tier either way. But a rising rate is the earliest sign that the receiver is degrading:
  a changed library or quantisation, or a shift in inputs. It also costs a re-send and GPU time on every
  occurrence.
- **The fallback alert knows the deployment mode.** The API exports `agent_receiver_mode{mode=...}`. In a replay or
  offline deployment (the live demo), answers from the rule are expected; with `RECEIVER=http`, the same rate means the
  GPU service is down.

The drift metrics come from the perception agent's own quality screen, so they cost nothing extra:

- `perception_input_noise`: a histogram of each recording's median above-40 Hz energy share;
- `perception_windows_total{readable}`;
- `perception_beats_total{label}`: the predicted class mix, to watch for label drift.

Grafana shows them in its input-noise, unreadable-share and beat-class-mix panels.

**Tested like code.** `ops/alerts_test.yml` feeds synthetic metric histories to `promtool`. It checks that each alert
fires with the right message, and that it stays quiet at the baseline, at low volume, and in replay mode. CI runs
`promtool check rules` and `promtool test rules` on every push:

```bash
docker run --rm -v "$PWD/ops:/ops" --entrypoint promtool prom/prometheus:v3.5.0 test rules /ops/alerts_test.yml
```

Alerts are evaluated by Prometheus (`docker compose --profile monitoring up`). Firing ones show at
http://localhost:9090/alerts. Routing them to email or Slack is a separate Alertmanager deployment, which is left out
here because nobody is on call for a demo.
