"""Baseline inference benchmark for the receiver (Gemma 4 E4B), before any optimisation (design §6d).

What it measures, per channel (compact / filtered / adapter) and window size (10 / 20 / 50 beats), batch size 1,
full constrained generation exactly as the agent runs it today:

  cold start      process start -> model loaded -> token table built -> warm-up done
  prompt_tokens   what the receiver reads
  ttft_ms         time to first token: the prompt has been processed (prefill)
  decision_ms     time until the tier token exists (what ranking needs)
  total_ms        the full answer with justification (what the agent waits for today)
  decode_tok_s    answer tokens per second after the first
  energy_j        GPU energy for the call (NVML counter)
  peak_mb         peak GPU memory allocated during the call
  agrees          receiver tier == the sender's rule tier (a sanity check, not the paper's accuracy figure)

Windows: from the bundled MIT-BIH test records, per size 5 windows chosen deterministically (2 urgent, 2 priority,
1 routine by the rule, across records). Resumable: results are saved after every call.

Run from backend/ with a CUDA environment that has transformers + bitsandbytes:
    python scripts/bench_inference.py                    # -> results/bench_baseline_<gpu>.json
    python scripts/bench_inference.py --summary          # print the table from the saved file
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

T_PROCESS = time.perf_counter()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from ecg_agent.core.models import deployed_adapter  # noqa: E402
from ecg_agent.core.rule import window_tier  # noqa: E402
from ecg_agent.core.sender import Sender, file_sha256  # noqa: E402
from ecg_agent.receiver.base import TriageRequest  # noqa: E402
from ecg_agent.signal.beats import from_mitdb  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ["100", "105", "200", "210", "213", "222", "233"]
SIZES = (10, 20, 50)
CHANNELS = ("compact", "filtered", "adapter")
WANT = {"urgent": 2, "priority": 2, "routine": 1}
SENDER_CK = ROOT / "artifacts/models/cnn_lstm_rr_seed0.pt"
ADAPTER_CK = deployed_adapter(ROOT / "artifacts/models")[0]


def save_atomic(path: Path, obj) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    tmp.replace(path)


def pick_windows() -> list[dict]:
    """Deterministic: walk the records in order, non-overlapping windows from the first 10 minutes."""
    sender = Sender(SENDER_CK, device="cpu")
    per_record = {}
    for rec in RECORDS:
        b = from_mitdb(ROOT / "data/records" / rec, 0, 600)
        o = sender.process(b.windows, b.rr_ms)
        per_record[rec] = (o.events, o.vectors)
    out = []
    for n in SIZES:
        need = dict(WANT)
        for k in range(200):  # round-robin over records so one patient does not dominate
            rec = RECORDS[k % len(RECORDS)]
            ev, vec = per_record[rec]
            start = (k // len(RECORDS)) * n
            if start + n > len(ev):
                continue
            tier = window_tier(ev[start:start + n])
            if need.get(tier, 0) > 0:
                need[tier] -= 1
                out.append({"id": f"{rec}:{start}:{n}", "record": rec, "start": start, "n": n, "rule_tier": tier,
                            "events": ev[start:start + n], "vectors": vec[start:start + n].tolist()})
            if not any(need.values()):
                break
    return out


def summarise(rows: list[dict]) -> dict:
    out = {}
    for ch in CHANNELS:
        for n in SIZES:
            xs = [r for r in rows if r["channel"] == ch and r["n"] == n]
            if not xs:
                continue
            med = lambda k, xs=xs: round(statistics.median(r[k] for r in xs if r[k] is not None), 1)  # noqa: E731
            out[f"{ch}/{n}"] = {"calls": len(xs), "prompt_tokens": med("prompt_tokens"), "ttft_ms": med("ttft_ms"),
                                "decision_ms": med("decision_ms"), "total_ms": med("total_ms"),
                                "gen_tokens": med("gen_tokens"), "decode_tok_s": med("decode_tok_s"),
                                "energy_j": med("energy_j"), "peak_mb": med("peak_mb"),
                                "agree": f"{sum(r['agrees'] for r in xs)}/{len(xs)}"}
    return out


def print_table(summary: dict, cold: dict) -> None:
    print(f"\nCold start: {json.dumps(cold)}")
    cols = ("prompt_tokens", "ttft_ms", "decision_ms", "total_ms", "gen_tokens", "decode_tok_s", "energy_j",
            "peak_mb", "agree")
    print(f"\n{'channel/N':14s}" + "".join(f"{c:>14s}" for c in cols))
    for k, v in summary.items():
        print(f"{k:14s}" + "".join(f"{str(v[c]):>14s}" for c in cols))


def decide_explain(engine, windows: list[dict], baseline: dict, out: Path) -> None:
    """Decide early, explain later, on the baseline's windows: per call, the decision (stop at the tier) and the
    explanation (resume after the tier), compared with the baseline's full answer for the same window and channel."""
    import pynvml

    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    base = {(r["window"], r["channel"]): r for r in baseline["rows"]}
    state = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {
        "design": "decide (stop at tier) then explain (resume after tier) vs the baseline full answer", "rows": []}
    done = {(r["window"], r["channel"]) for r in state["rows"]}
    todo = [(w, ch) for w in windows for ch in CHANNELS if (w["id"], ch) not in done]
    for i, (w, ch) in enumerate(todo):
        b = base[(w["id"], ch)]
        req = TriageRequest(channel=ch, events=w["events"], vectors=w["vectors"] if ch == "adapter" else None,
                            mode="decision")
        e0 = pynvml.nvmlDeviceGetTotalEnergyConsumption(h)
        d = engine.triage(req)
        e1 = pynvml.nvmlDeviceGetTotalEnergyConsumption(h)
        x = engine.explain(req, d.tier) if d.tier else None
        e2 = pynvml.nvmlDeviceGetTotalEnergyConsumption(h)
        overlap = min(len(b["answer"]), len(x.raw)) if x else 0  # the baseline stored the first 400 characters
        state["rows"].append({
            "window": w["id"], "n": w["n"], "channel": ch, "tier": d.tier, "baseline_tier": b["tier"],
            "same_tier": d.tier == b["tier"], "decide_ms": d.latency_ms, "decide_tokens": d.generated_tokens,
            "decide_energy_j": (e1 - e0) / 1e3, "explain_ms": x.latency_ms if x else None,
            "explain_tokens": x.generated_tokens if x else None, "explain_energy_j": (e2 - e1) / 1e3,
            "baseline_total_ms": b["total_ms"], "baseline_energy_j": b["energy_j"],
            "same_text": bool(x) and x.raw[:overlap] == b["answer"][:overlap], "answer": x.raw[:400] if x else ""})
        save_atomic(out, state)
        r = state["rows"][-1]
        print(f"[{i + 1}/{len(todo)}] {w['id']:>12s} {ch:8s} tier={d.tier} (baseline {b['tier']}) "
              f"decide={r['decide_ms']:.0f}ms explain={r['explain_ms'] or 0:.0f}ms baseline={b['total_ms']:.0f}ms "
              f"same_text={r['same_text']}", flush=True)
    rows = state["rows"]
    med = lambda xs: round(statistics.median(xs), 1)  # noqa: E731
    state["summary"] = {
        "calls": len(rows), "same_tier": sum(r["same_tier"] for r in rows), "same_text": sum(r["same_text"] for r in rows),
        "by_channel": {ch: {"decide_ms": med([r["decide_ms"] for r in rows if r["channel"] == ch]),
                            "explain_ms": med([r["explain_ms"] for r in rows if r["channel"] == ch and r["explain_ms"]]),
                            "baseline_total_ms": med([r["baseline_total_ms"] for r in rows if r["channel"] == ch]),
                            "decide_energy_j": med([r["decide_energy_j"] for r in rows if r["channel"] == ch]),
                            "baseline_energy_j": med([r["baseline_energy_j"] for r in rows if r["channel"] == ch])}
                       for ch in CHANNELS}}
    save_atomic(out, state)
    print(json.dumps(state["summary"], indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--decide-explain", action="store_true", help="time decide + explain against the baseline")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    out = args.out or ROOT / "results" / f"bench_baseline_{gpu.split()[-1].lower()}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    state = json.loads(out.read_text(encoding="utf-8")) if out.exists() else None
    if state and not args.summary and state.get("adapter_sha256") != file_sha256(ADAPTER_CK):
        sys.exit(f"{out.name} was measured with another adapter; move it aside to benchmark the deployed one")
    if args.summary:
        print_table(state["summary"], state["cold_start"])
        return

    import pynvml

    from ecg_agent.receiver.gemma import MODEL_ID, GemmaEngine

    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    t_import = time.perf_counter()
    windows = pick_windows()
    t_windows = time.perf_counter()
    engine = GemmaEngine(ADAPTER_CK)  # load + token table + warm-up
    t_ready = time.perf_counter()
    cold = {"imports_s": round(t_import - T_PROCESS, 1), "engine_ready_s": round(t_ready - t_windows, 1),
            "gpu_memory_after_load_mb": round(torch.cuda.memory_allocated() / 2**20)}
    if args.decide_explain:
        decide_explain(engine, windows, state, out.with_name(out.stem.replace("baseline", "decide_explain") + ".json"))
        return
    if state is None:
        state = {"design": __doc__, "gpu": gpu, "model": MODEL_ID, "quantisation": "bitsandbytes nf4",
                 "torch": torch.__version__, "platform": platform.platform(),
                 "sender_sha256": file_sha256(SENDER_CK), "adapter_sha256": file_sha256(ADAPTER_CK),
                 "windows": [{k: w[k] for k in ("id", "record", "start", "n", "rule_tier")} for w in windows],
                 "cold_start": cold, "rows": []}
    done = {(r["window"], r["channel"]) for r in state["rows"]}
    todo = [(w, ch) for w in windows for ch in CHANNELS if (w["id"], ch) not in done]
    print(f"{len(todo)} calls to run ({len(done)} already done); engine ready in {cold['engine_ready_s']} s",
          flush=True)

    for i, (w, ch) in enumerate(todo):
        req = TriageRequest(channel=ch, events=w["events"], vectors=w["vectors"] if ch == "adapter" else None)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        e0 = pynvml.nvmlDeviceGetTotalEnergyConsumption(h)
        res = engine.triage(req)
        e1 = pynvml.nvmlDeviceGetTotalEnergyConsumption(h)
        n_gen = res.generated_tokens or 0
        decode = (res.latency_ms - res.ttft_ms) / 1e3 if res.latency_ms and res.ttft_ms else None
        state["rows"].append({
            "window": w["id"], "n": w["n"], "rule_tier": w["rule_tier"], "channel": ch, "tier": res.tier,
            "parsed": res.parsed, "agrees": res.tier == w["rule_tier"], "prompt_tokens": res.prompt_tokens,
            "ttft_ms": res.ttft_ms, "decision_ms": res.decision_ms, "total_ms": res.latency_ms,
            "gen_tokens": n_gen, "decode_tok_s": (n_gen - 1) / decode if decode and n_gen > 1 else None,
            "energy_j": (e1 - e0) / 1e3, "peak_mb": torch.cuda.max_memory_allocated() / 2**20,
            "answer": res.raw[:400]})
        state["summary"] = summarise(state["rows"])
        save_atomic(out, state)
        r = state["rows"][-1]
        print(f"[{i + 1}/{len(todo)}] {w['id']:>12s} {ch:8s} rule={w['rule_tier']:8s} got={str(res.tier):8s} "
              f"prompt={r['prompt_tokens']} ttft={r['ttft_ms']:.0f}ms decision={r['decision_ms']:.0f}ms "
              f"total={r['total_ms']:.0f}ms gen={n_gen}", flush=True)
    print_table(state["summary"], state["cold_start"])


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    np.set_printoptions(precision=3)
    main()
