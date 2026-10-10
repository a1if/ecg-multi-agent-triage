"""Promotion gate: may a challenger adapter replace the champion? Decided on held-out patients, with rules fixed in
advance, so a model change ships because the evidence says so and not because one number looked higher.

    python scripts/promotion_gate.py --extract PAPER_REPO   # (once) copy the per-window held-out answers into results/
    python scripts/promotion_gate.py                        # decide; writes results/promotion_gate.json
    python scripts/promotion_gate.py --check                # exit 1 if the deployed adapter is not the gate's choice
    python scripts/promotion_gate.py --mlflow               # also log the decision to MLflow (evaluation experiment)

Data: the paper's held-out evaluation (MIT-BIH DS2: 22 patients never seen in training), where every adapter seed
answered the same windows through the same Gemma 4 E4B with constrained decoding. Seeds are compared on identical
windows (paired), so differences in which windows were drawn cannot favour one seed.

Rules (fixed before any confidence interval was computed):
  set          the stratified set (balanced across tiers: the designed test); the natural set is reported, not gated
  uncertainty  paired bootstrap over patients (records), not windows: windows from one patient are correlated, and
               treating them as independent would make the intervals look tighter than they are. 10,000 resamples.
  multiplicity two challengers are compared with the champion, so each interval is 97.5% (Bonferroni, 0.05 / 2)
  promote if   1. balanced accuracy: the lower bound of (challenger - champion) is above 0
               2. urgent recall: the lower bound of (challenger - champion) is above -0.05 (non-inferiority: a
                  challenger may not be meaningfully worse at recognising urgent windows, whatever else it gains)
               3. parse rate: not lower than the champion's
  if several pass, the highest balanced accuracy wins
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "heldout_adapter_seeds.json"
OUT = ROOT / "results" / "promotion_gate.json"
MANIFEST = ROOT / "artifacts" / "models" / "manifest.json"
TIERS = ("routine", "priority", "urgent")
RESAMPLES, SEED, ALPHA, URGENT_MARGIN = 10_000, 20261010, 0.05, 0.05
FIELDS = ("arm", "set", "n", "start", "record", "reference", "tier", "parsed")


def extract(paper: Path) -> None:
    src = paper / "results" / "p1_item7_eval_ds2v2_r4.json"
    d = json.loads(src.read_text())
    rows = [{k: r[k] for k in FIELDS} for r in d["rows"] if r["arm"].startswith("MEA:")]
    files = {f"seed{s}": hashlib.sha256((paper / "reasoning" / "checkpoints" / f"p1_item7_mea_r4_seed{s}.pt")
                                        .read_bytes()).hexdigest() for s in (101, 202, 303)}
    DATA.write_text(json.dumps({"source": f"{src.name} (paper repo)", "adapter_sha256": files, "rows": rows}))
    print(f"wrote {DATA.relative_to(ROOT)}: {len(rows)} answers")


def metrics(ref: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    """Balanced accuracy (mean recall over the three tiers) and urgent recall."""
    recalls = [np.mean(pred[ref == t] == t) for t in TIERS]
    return float(np.mean(recalls)), float(recalls[2])


def decide(data: dict) -> dict:
    rows = [r for r in data["rows"] if r["set"] == "stratified"]
    seeds = sorted({r["arm"].split("seed")[1] for r in rows})
    key = lambda r: (r["record"], r["n"], r["start"])  # noqa: E731
    by_seed = {s: {key(r): r for r in rows if r["arm"].endswith(f"seed{s}")} for s in seeds}
    windows = sorted(set.intersection(*(set(v) for v in by_seed.values())))  # identical windows for every seed
    records = np.array([w[0] for w in windows])
    ref = np.array([by_seed[seeds[0]][w]["reference"] for w in windows])
    pred = {s: np.array([by_seed[s][w]["tier"] or "" for w in windows]) for s in seeds}
    parsed = {s: float(np.mean([by_seed[s][w]["parsed"] for w in windows])) for s in seeds}

    manifest = json.loads(MANIFEST.read_text())
    champion = next(s for s, h in data["adapter_sha256"].items() if h == manifest["adapter"]["sha256"]).removeprefix("seed")
    challengers = [s for s in seeds if s != champion]
    level = 1 - ALPHA / len(challengers)

    # Paired cluster bootstrap: draw patients with replacement, take all their windows, score every seed on them.
    rng = np.random.default_rng(SEED)
    patients = np.unique(records)
    idx = {p: np.flatnonzero(records == p) for p in patients}
    boot = {s: [] for s in seeds}
    for _ in range(RESAMPLES):
        take = np.concatenate([idx[p] for p in rng.choice(patients, size=len(patients), replace=True)])
        for s in seeds:
            boot[s].append(metrics(ref[take], pred[s][take]))
    boot = {s: np.array(v) for s, v in boot.items()}
    lo, hi = (1 - level) / 2 * 100, (1 + level) / 2 * 100

    point = {s: metrics(ref, pred[s]) for s in seeds}
    result = {"rules": __doc__.split("Rules")[1].strip(), "n_windows": len(windows), "n_patients": len(patients),
              "champion": f"seed{champion}", "interval": f"{level:.1%}", "seeds": {}, "comparisons": {}}
    for s in seeds:
        result["seeds"][f"seed{s}"] = {"balanced_accuracy": round(point[s][0], 4), "urgent_recall": round(point[s][1], 4),
                                       "parse_rate": parsed[s]}
    passing = []
    for s in challengers:
        diff = boot[s] - boot[champion]
        ba = [float(np.percentile(diff[:, 0], q)) for q in (lo, hi)]
        ur = [float(np.percentile(diff[:, 1], q)) for q in (lo, hi)]
        checks = {"balanced_accuracy_better": ba[0] > 0, "urgent_recall_not_worse": ur[0] > -URGENT_MARGIN,
                  "parse_rate_not_lower": parsed[s] >= parsed[champion]}
        result["comparisons"][f"seed{s}"] = {
            "balanced_accuracy_diff": round(point[s][0] - point[champion][0], 4),
            "balanced_accuracy_ci": [round(x, 4) for x in ba],
            "urgent_recall_diff": round(point[s][1] - point[champion][1], 4),
            "urgent_recall_ci": [round(x, 4) for x in ur],
            "checks": checks, "passes": all(checks.values())}
        if all(checks.values()):
            passing.append(s)
    winner = max(passing, key=lambda s: point[s][0]) if passing else champion
    result["decision"] = f"promote seed{winner}" if passing else f"keep seed{champion}"
    result["selected"] = f"seed{winner}"
    result["selected_sha256"] = data["adapter_sha256"][f"seed{winner}"]

    # The natural set (real class mix) for context only.
    result["natural_set"] = {}
    for s in seeds:
        nat = [r for r in data["rows"] if r["set"] == "natural" and r["arm"].endswith(f"seed{s}")]
        ba = metrics(np.array([r["reference"] for r in nat]), np.array([r["tier"] or "" for r in nat]))[0]
        result["natural_set"][f"seed{s}"] = round(ba, 4)
    return result


def report(r: dict) -> None:
    print(f"{r['n_windows']} held-out windows from {r['n_patients']} patients; champion {r['champion']}; "
          f"{r['interval']} intervals, bootstrap over patients\n")
    print(f"{'seed':<9}{'bal. acc':>9}{'urgent rec':>12}{'parsed':>8}")
    for s, m in r["seeds"].items():
        print(f"{s:<9}{m['balanced_accuracy']:>9.3f}{m['urgent_recall']:>12.3f}{m['parse_rate']:>8.0%}")
    print()
    for s, c in r["comparisons"].items():
        ba, ur = c["balanced_accuracy_ci"], c["urgent_recall_ci"]
        print(f"{s} vs {r['champion']}: balanced accuracy {c['balanced_accuracy_diff']:+.3f} [{ba[0]:+.3f}, {ba[1]:+.3f}]  "
              f"urgent recall {c['urgent_recall_diff']:+.3f} [{ur[0]:+.3f}, {ur[1]:+.3f}]  "
              f"-> {'PASS' if c['passes'] else 'fail: ' + ', '.join(k for k, v in c['checks'].items() if not v)}")
    print(f"\ndecision: {r['decision']}")


def log_mlflow(r: dict) -> None:
    import mlflow

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    mlflow.set_experiment("evaluation")
    with mlflow.start_run(run_name="promotion_gate"):
        mlflow.set_tags({"result_file": OUT.name, "decision": r["decision"], "champion": r["champion"]})
        for s, m in r["seeds"].items():
            mlflow.log_metrics({f"{s}.{k}": v for k, v in m.items()})
        for s, c in r["comparisons"].items():
            mlflow.log_metrics({f"{s}.balanced_accuracy_ci_low": c["balanced_accuracy_ci"][0],
                                f"{s}.urgent_recall_ci_low": c["urgent_recall_ci"][0]})
        mlflow.log_artifact(str(OUT), artifact_path="result")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", type=Path, metavar="PAPER_REPO")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--mlflow", action="store_true")
    a = ap.parse_args()
    if a.extract:
        extract(a.extract)
    res = decide(json.loads(DATA.read_text()))
    if a.check:
        deployed = json.loads(MANIFEST.read_text())["adapter"]["sha256"]
        ok = deployed == res["selected_sha256"]
        print(f"deployed adapter {'is' if ok else 'is NOT'} the gate's choice ({res['decision']})")
        sys.exit(0 if ok else 1)
    OUT.write_text(json.dumps(res, indent=1))
    report(res)
    if a.mlflow:
        log_mlflow(res)
