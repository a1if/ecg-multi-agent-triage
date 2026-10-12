"""Record real Gemma answers for the public demo's presets, through a running API in ``http`` mode.

The public demo runs without a GPU (RECEIVER=replay): it serves answers recorded here, keyed by exactly what Gemma
saw, and labels them "Gemma (recorded answer)". This script drives every preset run, every explanation and a set of
typical questions through the API, so the API's replay store records them; then copy the store into the image:

    docker compose cp api:/app/data/replay/answers.jsonl backend/data/replay/answers.jsonl

Then check the recording is complete as deployed (API with RECEIVER=replay, built from the same pinned image):

    python scripts/record_demo_answers.py --verify     # exit 1 if any preset answer is not served from the recording

A miss means the demo would quietly answer from the rule: a dependency changed the numbers, or a preset is missing.

Presets (must match the API's demo mode, ecg_agent/api/settings.py DEMO_*): each bundled record, 0-300 s,
6 reviews, every mode, no scenario and **every scenario the API offers** (taken from /v1/scenarios, so a new one is
never left unrecorded). Scenarios that make a recording unreadable cost nothing: no LLM is asked.
"""
from __future__ import annotations

import argparse
import sys
import time

import httpx

RECORDS = ["100", "105", "200", "210", "213", "222", "233"]
MODES = ["balanced", "accuracy", "throughput"]


def wait(c: httpx.Client, rid: str) -> dict:
    while True:
        d = c.get(f"/v1/runs/{rid}").json()
        if d["status"] not in ("queued", "running"):
            return d
        time.sleep(1)


def misses(c: httpx.Client) -> int:
    return c.get("/v1/status").json()["receiver_health"].get("replay_misses", 0)


def demonstrable(d: dict, explanations: dict, comparisons: dict, answers: list[dict]) -> list[str]:
    """What a visitor would see go wrong in this combination (empty: it demonstrates as designed)."""
    if d["status"] == "unreadable":
        return []  # correct outcome: no tier, sent to a person, no LLM involved
    if d["status"] != "complete":
        return [f"run {d['status']}"]
    problems = []
    if not d["findings"]:
        problems.append("no window reviewed")
    for wid, e in explanations.items():
        if not e.get("justification"):
            problems.append(f"{wid}: no explanation")
    for wid, cmp in comparisons.items():
        for ch in ("adapter", "filtered"):
            m = cmp.get(ch) or {}
            if not m.get("note") and (m.get("prompt_tokens") is None or m.get("energy_j") is None):
                problems.append(f"{wid}: {ch} not measured")
    for a in answers:
        if not a.get("text"):
            problems.append(f"no answer to {a.get('question')!r}")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--records", nargs="*", default=RECORDS)
    ap.add_argument("--verify", action="store_true",
                    help="replay mode: check every combination is served from the recording and demonstrates fully")
    a = ap.parse_args()
    failures: list[str] = []
    t0, n = time.time(), 0
    with httpx.Client(base_url=a.api, timeout=600) as c:
        st = c.get("/v1/status").json()
        if a.verify and st["receiver"] != "replay":
            sys.exit(f"--verify needs an API with RECEIVER=replay; got {st['receiver']}")
        if not a.verify and (st["receiver"] != "http"
                             or st["receiver_health"].get("live", {}).get("state") != "ready"):
            sys.exit(f"the API must use a ready GPU service (RECEIVER=http); got {st['receiver_health']}")
        seen = misses(c)
        scenarios = [None] + c.get("/v1/scenarios").json()  # exactly what the demo's dropdown offers
        for rec in a.records:
            for scen in scenarios:
                for mode in MODES:
                    body = {"record": rec, "start_s": 0, "duration_s": 300, "mode": mode, "scenario": scen,
                            "max_reviews": 6}
                    d = wait(c, c.post("/v1/runs", json=body).json()["id"])
                    run = f"/v1/runs/{d['id']}"
                    explanations, comparisons = {}, {}
                    for wid in sorted(d["findings"]):
                        explanations[wid] = c.post(f"{run}/windows/{wid}/explain").json()
                        comparisons[wid] = c.post(f"{run}/windows/{wid}/compare").json()  # "Adapter vs text"
                    # The suggested questions, in the order the Ask tab offers them (same wording, from the API).
                    for q in d.get("suggestions", []):
                        c.post(f"{run}/questions", json={"question": q})
                    answers = c.get(run).json()["answers"]
                    srcs = sorted({t["source"] for x in d["findings"].values() for t in x["attempts"]})
                    # Ask the store itself: any request it could not answer (triage, explanation, question routing,
                    # summary) counts, whatever the agents then did with the fallback.
                    now, n = misses(c), n + 1
                    where = f"{rec} {mode} {scen or '-'}"
                    if a.verify:
                        if now > seen:
                            failures.append(f"{where}: {now - seen} request(s) not in the recording")
                        failures += [f"{where}: {p}" for p in demonstrable(d, explanations, comparisons, answers)]
                    seen = now
                    print(f"{time.time() - t0:6.0f}s {rec} {mode:10s} {scen or '-':11s} {d['status']:10s} "
                          f"windows={len(d['findings'])} sources={srcs} questions={len(answers)}", flush=True)
        print(f"done in {time.time() - t0:.0f} s; replay entries now: "
              f"{c.get('/v1/status').json()['receiver_health'].get('replay_entries')}")
    if a.verify:
        print("\n".join(failures) or f"all {n} combinations demonstrate fully from the recording (0 misses)")
        sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
