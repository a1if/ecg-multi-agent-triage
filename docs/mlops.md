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
| registry `ecg-adapter` | The virtual-token adapter: seed 101 (v1, champion), seeds 202 and 303 (v2, v3, challengers) |
| experiment `evaluation` | One run per result file in `backend/results/` (system evaluation, stress, real-Gemma, inference benchmarks, chaos), each tagged with the model versions and SHA-256s it tested |

**How it ties to deployment.** The services don't load from MLflow. They load the files in
`backend/artifacts/models/`, pinned by SHA-256 in `manifest.json`, and the orchestrator refuses virtual tokens from a
sender whose SHA-256 differs from the one the adapter was trained for. The registry is the record around those files.
The `champion` alias is set to the version whose SHA-256 is in the manifest, and `--check` fails if the two ever
disagree. It can run as a gate before a deploy.

Why not serve straight from the registry? A demo with no paid infrastructure can't depend on a tracking server being
up, and pinning by content hash means every image is reproducible on its own. In a team setting the next step would be
a CI job that downloads `models:/ecg-adapter@champion`, checks its SHA-256 against the manifest and builds the image.

### A finding: the champion is not the best validation score

| Adapter | Validation balanced accuracy |
|---|---|
| seed 101 (deployed) | 0.719 |
| seed 202 | 0.799 |
| seed 303 | 0.819 |

The deployed adapter scores lowest on the validation split. It stays champion for now, and the reason is stored on the
version as the `champion_reason` tag:

- The validation split is small (about 30 windows per size), so differences of this size are within noise.
- Every downstream measurement uses seed 101: the paper's results, the inference benchmarks and the 257 recorded Gemma
  answers behind the live demo.

Promoting a challenger takes three steps. Evaluate it on the held-out test windows, not validation. If it wins there,
re-record the demo's answers. Then move the `champion` alias and the manifest together. Changing a model in a clinical
pipeline means re-checking everything measured with the old one, and this is that rule written down.

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
