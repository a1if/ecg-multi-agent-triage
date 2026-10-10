"""Run both agents end to end with real Gemma on this machine's GPU, then explain every finding.

    python scripts/run_agent_local.py --record 105 --minutes 5 --mode throughput --budget 3
    python scripts/run_agent_local.py --planner gemma          # Gemma also plans (slower: ~25 s per planning step)

Prints the conversation, the report and each explanation, and saves the full trace to results/agent_run_<...>.json.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from ecg_agent.agent.harness import Budget, Tracer
from ecg_agent.agent.orchestrator import Orchestrator
from ecg_agent.agent.perception_agent import PerceptionAgent, load_mitdb
from ecg_agent.agent.planner import GemmaPlanner, RulePlanner
from ecg_agent.agent.policy import Policy
from ecg_agent.agent.questions import GemmaClassifier
from ecg_agent.agent.reasoning_agent import ReasoningAgent
from ecg_agent.core.models import deployed_adapter
from ecg_agent.core.sender import Sender
from ecg_agent.receiver.gemma import GemmaEngine
from ecg_agent.receiver.local import LocalGemmaReceiver

ROOT = Path(__file__).resolve().parents[1]


def show(step: dict) -> None:
    d = step["data"]
    if step["kind"] == "message":
        m = d["message"]
        c = {k: v for k, v in m["content"].items() if k in ("window_id", "channel", "purpose", "reason")}
        print(f"  {m['id']} {m['from']:>10} -> {m['to']:<12} {m['intent']:15} {json.dumps(c)[:200]}")
    else:
        print(f"{step['agent']:>12} [{step['kind']}] {step['title']}")


async def main(args) -> None:
    t0 = time.perf_counter()
    engine = GemmaEngine(deployed_adapter(ROOT / "artifacts/models")[0])
    print(f"Gemma ready in {time.perf_counter() - t0:.0f} s")
    receiver = LocalGemmaReceiver(engine)
    tracer, budget = Tracer(f"local-{args.record}"), Budget(max_reviews=args.budget)
    sender = Sender(ROOT / "artifacts/models/cnn_lstm_rr_seed0.pt")
    p = PerceptionAgent(sender, load_mitdb(ROOT / "data/records", args.record, 0, 60 * args.minutes), tracer,
                        args.mode, budget)
    planner = GemmaPlanner(receiver) if args.planner == "gemma" else RulePlanner()
    r = ReasoningAgent(receiver, planner, tracer, args.mode, budget, classifier=GemmaClassifier(receiver))
    policy = Policy.from_manifest(ROOT / "artifacts/models/manifest.json")
    orch = Orchestrator(p, r, tracer, policy, budget,
                        audit_path=ROOT / "results" / f"audit_{args.record}_{args.mode}_{args.planner}.jsonl")

    t1 = time.perf_counter()
    report = await orch.run()
    run_s = time.perf_counter() - t1
    for s in tracer.steps:
        show(s)
    print(f"\nRun: {run_s:.1f} s | overall {report['overall_tier']} | human review {report['needs_human_review']}")
    print(f"Narrative ({report['narrative_source']}): {report['narrative']}\n")

    for q in args.ask:
        n0, t2 = len(tracer.steps), time.perf_counter()
        a = await orch.ask(q)
        kind = next((s["title"] for s in tracer.steps[n0:] if s["title"].startswith("Question type")), "screened")
        print(f"Q: {q}\n   -> {kind} ({time.perf_counter() - t2:.1f} s)\nA [{a['type']}]: {a['text']}\n")

    for wid in ([] if args.ask else sorted(r.findings)):
        n0 = len(tracer.steps)
        t2 = time.perf_counter()
        exp = await orch.explain(wid)
        f = r.findings[wid]
        print(f"--- explain {wid} (final {f['final_tier']}, decided over {f['attempts'][-1]['channel']}, "
              f"{time.perf_counter() - t2:.1f} s)")
        for s in tracer.steps[n0:]:
            show(s)
        print(f"    by {exp['explained_by']}{' | ' + exp['note'] if exp.get('note') else ''}\n"
              f"    {exp['justification']}\n")

    out = ROOT / "results" / f"agent_run_{args.record}_{args.mode}_{args.planner}.json"
    out.write_text(json.dumps({"args": vars(args), "run_s": run_s, "report": report, "steps": tracer.steps,
                               "findings": r.findings}, indent=1, default=str), encoding="utf-8")
    print(f"Saved {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", default="105")
    ap.add_argument("--minutes", type=float, default=5)
    ap.add_argument("--mode", choices=("accuracy", "balanced", "throughput"), default="throughput")
    ap.add_argument("--budget", type=int, default=3)
    ap.add_argument("--planner", choices=("rule", "gemma"), default="rule")
    ap.add_argument("--ask", nargs="*", default=[], help="clinician questions to ask after the run")
    asyncio.run(main(ap.parse_args()))
