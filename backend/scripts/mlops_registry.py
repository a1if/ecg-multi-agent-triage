"""MLflow: register the models with their lineage, log the evaluations against them, and check deployment matches.

    docker compose --profile mlops up -d mlflow        # tracking server + registry UI at http://localhost:5000
    python scripts/mlops_registry.py                   # (re)build: models, versions, champion, evaluation runs
    python scripts/mlops_registry.py --paper-repo PATH # also register the paper's other adapter seeds as challengers
    python scripts/mlops_registry.py --check           # exit 1 if the deployed manifest is not the registry champion

What goes in:
  experiment "training"    one run per model file: training metrics (from the checkpoint, and from the paper repo's
                           results when available), parameters, SHA-256, and for the adapter the sender it was
                           trained for; each run's checkpoint is registered as a model version
  registry                 ecg-sender and ecg-adapter; alias "champion" = the version whose SHA-256 is in
                           artifacts/models/manifest.json (what the services load), with the reason recorded
  experiment "evaluation"  the system evaluation, signal-fault, real-Gemma, inference and chaos results in
                           results/, each tagged with the model versions it tested (lineage: result -> model)

The manifest stays what deployment pins (by SHA-256); the registry is the record of how each version came to be,
how it was evaluated, and why the champion is the champion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import mlflow
import torch
from mlflow import MlflowClient

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "artifacts" / "models"
RESULTS = ROOT / "results"
PAPER = "https://github.com/a1if/Heterogeneous-Multi-Agent-Edge-AI-for-Clinical-Decision-Support"
CHAMPION_REASON = ("Promoted from seed 101 by scripts/promotion_gate.py on held-out patients (MIT-BIH DS2, 795 windows, "
                   "22 patients, paired bootstrap over patients): balanced accuracy +0.050 [+0.020, +0.079] and urgent "
                   "recall not worse. Seed 202 failed the gate (urgent recall -0.083). Demo answers re-recorded with it.")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def flat(prefix: str, d: dict) -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flat(key, v))
        elif isinstance(v, int | float) and not isinstance(v, bool):
            out[key] = float(v)
    return out


def register(client: MlflowClient, name: str, path: Path, params: dict, metrics: dict, tags: dict,
             description: str) -> str:
    """One training-lineage run + one model version for a checkpoint file. Idempotent by SHA-256."""
    digest = sha256(path)
    for mv in client.search_model_versions(f"name='{name}'"):
        if mv.tags.get("sha256") == digest:
            print(f"  {name} v{mv.version} already registered ({path.name})")
            return mv.version
    mlflow.set_experiment("training")
    with mlflow.start_run(run_name=path.stem) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.set_tags({**tags, "sha256": digest, "file": path.name, "source": PAPER})
        mlflow.log_artifact(str(path), artifact_path="checkpoint")
    # A raw checkpoint file, not an MLflow-format model: register the run's artifact directly as the version source.
    mv = client.create_model_version(name, f"{run.info.artifact_uri}/checkpoint/{path.name}", run_id=run.info.run_id,
                                     tags={**tags, "sha256": digest, "file": path.name}, description=description)
    print(f"  {name} v{mv.version} <- {path.name}")
    return mv.version


def build(paper_repo: Path | None) -> None:
    client = MlflowClient()
    manifest = json.loads((MODELS / "manifest.json").read_text())
    for name, desc in (("ecg-sender", "Perception agent's CNN-LSTM with RR-interval branch (sender)."),
                       ("ecg-adapter", "Multi-event adapter: sender vectors -> 4 virtual tokens per beat for Gemma 4 E4B.")):
        if not client.search_registered_models(f"name='{name}'"):
            client.create_registered_model(name, description=desc,
                                           tags={"project": "ecg-multi-agent-triage", "source": PAPER})

    # ---- sender ----
    sp = MODELS / manifest["sender"]["file"]
    ck = torch.load(sp, map_location="cpu", weights_only=False)
    metrics, params = {}, {"seed": ck.get("seed"), "arch": manifest["sender"]["arch"],
                           "rr_features": ",".join(ck.get("rr_feature_names", [])), "context_dim": 32}
    res = paper_repo / "results" / "p1_rr_encoder_results.json" if paper_repo else None
    if res and res.exists():
        d = json.loads(res.read_text())
        metrics.update(flat("ds2", d["ds2"]["rr_encoder"]))
        params["train_git_commit"] = d.get("git_commit", "")[:12]
    sender_v = register(client, "ecg-sender", sp, params, metrics, {"role": "champion"},
                        "Held-out patients (MIT-BIH DS2) metrics from the paper repo when available.")

    # ---- adapters: the deployed one, and the paper's other seeds as challengers ----
    adapter_files = [MODELS / manifest["adapter"]["file"]]
    if paper_repo:
        adapter_files += [p for p in sorted((paper_repo / "reasoning" / "checkpoints").glob("p1_item7_mea_r4_seed*.pt"))
                          if sha256(p) != manifest["adapter"]["sha256"]]
    adapter_versions, seed_versions = {}, {}
    for ap in adapter_files:
        ck = torch.load(ap, map_location="cpu", weights_only=False)
        sd = ck["adapter_state_dict"]
        deployed = sha256(ap) == manifest["adapter"]["sha256"]
        v = register(client, "ecg-adapter", ap,
                     {"seed": ck["seed"], "updates": ck["update"], "input_dim": int(sd["projection.weight"].shape[1]),
                      "max_events": int(sd["position"].shape[0]), "tokens_per_event": int(sd["position"].shape[1]),
                      "receiver": manifest["adapter"]["receiver"],
                      "quantisation": manifest["adapter"]["receiver_quantisation"]},
                     flat("val", ck["val"]),
                     {"role": "champion" if deployed else "challenger",
                      "trained_for_sender_sha256": manifest["adapter"]["trained_for_sender_sha256"]},
                     "Validation metrics from training (small split: about 30 windows per size).")
        adapter_versions[v] = deployed
        seed_versions[str(ck["seed"])] = v
        # Roles follow the manifest: after a promotion the old champion becomes a challenger on the next build.
        client.set_model_version_tag("ecg-adapter", v, "role", "champion" if deployed else "challenger")
        if not deployed and "champion_reason" in client.get_model_version("ecg-adapter", v).tags:
            client.delete_model_version_tag("ecg-adapter", v, "champion_reason")

    # ---- champion aliases: what the manifest (deployment) pins ----
    champ = next(v for v, d in adapter_versions.items() if d)
    client.set_registered_model_alias("ecg-adapter", "champion", champ)
    client.set_model_version_tag("ecg-adapter", champ, "champion_reason", CHAMPION_REASON)
    client.set_registered_model_alias("ecg-sender", "champion", sender_v)

    # ---- evaluations, linked to the versions they tested ----
    mlflow.set_experiment("evaluation")
    lineage = {"model.sender.version": sender_v, "model.sender.sha256": manifest["sender"]["sha256"]}
    # Skip by content, not name: a re-run of eval_gemma.json after a promotion is a new result under the same name.
    logged = {r.data.tags.get("result_sha256") for r in client.search_runs(
        [client.get_experiment_by_name("evaluation").experiment_id], max_results=1000)}
    for path in sorted(RESULTS.glob("*.json")):
        if sha256(path) in logged or not path.name.startswith(("eval_", "bench_", "chaos_", "promotion_")):
            continue
        # A result names the adapter seed it measured (e.g. bench_baseline_5070_seed101.json); otherwise it is the
        # current champion's. Linking it to the wrong version would make the history say something it does not.
        seed = re.search(r"seed(\d+)", path.stem)
        adapter_v = seed_versions.get(seed.group(1), champ) if seed else champ
        d = json.loads(path.read_text())
        m = {k: v for k, v in flat("", {k: v for k, v in d.items() if k not in ("rows", "runs", "questions", "attempts",
                                                                                  "windows")}).items()}
        with mlflow.start_run(run_name=path.stem):
            mlflow.set_tags({**lineage, "model.adapter.version": adapter_v,
                             "model.adapter.sha256": client.get_model_version("ecg-adapter", adapter_v).tags["sha256"],
                             "result_file": path.name, "result_sha256": sha256(path)})
            if m:
                mlflow.log_metrics({k.replace("/", "_")[:250]: v for k, v in m.items()})
            mlflow.log_artifact(str(path), artifact_path="result")
        print(f"  evaluation run <- {path.name} ({len(m)} metrics)")
    print(f"done: ecg-sender@champion = v{sender_v}, ecg-adapter@champion = v{champ}")


def check() -> int:
    """Deployment consistency: the manifest the services load must be the registry's champion versions."""
    client = MlflowClient()
    manifest = json.loads((MODELS / "manifest.json").read_text())
    ok = True
    for name, key in (("ecg-sender", "sender"), ("ecg-adapter", "adapter")):
        mv = client.get_model_version_by_alias(name, "champion")
        same = mv.tags.get("sha256") == manifest[key]["sha256"]
        ok &= same
        print(f"{name}@champion v{mv.version}: {'matches' if same else 'DOES NOT MATCH'} the deployed manifest")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper-repo", type=Path)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # MLflow prints emoji; the Windows console's default encoding cannot
    sys.stderr.reconfigure(encoding="utf-8")
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    sys.exit(check() if a.check else build(a.paper_repo))
