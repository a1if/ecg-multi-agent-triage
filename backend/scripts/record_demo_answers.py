"""Record real Gemma answers for the public demo's presets, through a running API in ``http`` mode.

The public demo runs without a GPU (RECEIVER=replay): it serves answers recorded here, keyed by exactly what Gemma
saw, and labels them "Gemma (recorded answer)". This script drives every preset run, every explanation and a set of
typical questions through the API, so the API's replay store records them; then copy the store into the image:

    docker compose cp api:/app/data/replay/answers.jsonl backend/data/replay/answers.jsonl

Then check the recording is complete as deployed (API with RECEIVER=replay, built from the same pinned image):

    python scripts/record_demo_answers.py --verify     # exit 1 if any preset answer is not served from the recording

A miss means the demo would quietly answer from the rule: a dependency changed the numbers, or a preset is missing.

Presets (must match the API's demo mode, ecg_agent/api/settings.py DEMO_*): each bundled record, 0-300 s,
6 reviews, every mode, no scenario or the noise scenarios.
"""
from __future__ import annotations

import argparse
import sys
import time

import httpx

RECORDS = ["100", "105", "200", "210", "213", "222", "233"]
MODES = ["balanced", "accuracy", "throughput"]
SCENARIOS = [None, "noise_burst"]  # noise_10db / clipped end as unreadable: no LLM involved
QUESTIONS = ["Why is {top} {tier}?", "Show me the beats in {top}", "Is there a run of abnormal beats near the start?",
             "Did the channels disagree anywhere?", "How does this system work?", "Can you look at {unreviewed}?"]


def wait(c: httpx.Client, rid: str) -> dict:
    while True:
        d = c.get(f"/v1/runs/{rid}").json()
        if d["status"] not in ("queued", "running"):
            return d
        time.sleep(1)


def misses(c: httpx.Client) -> int:
    return c.get("/v1/status").json()["receiver_health"].get("replay_misses", 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--records", nargs="*", default=RECORDS)
    ap.add_argument("--verify", action="store_true", help="replay mode: check every answer comes from the recording")
    a = ap.parse_args()
    missed: list[str] = []
    t0 = time.time()
    with httpx.Client(base_url=a.api, timeout=600) as c:
        st = c.get("/v1/status").json()
        if a.verify and st["receiver"] != "replay":
            sys.exit(f"--verify needs an API with RECEIVER=replay; got {st['receiver']}")
        if not a.verify and (st["receiver"] != "http"
                             or st["receiver_health"].get("live", {}).get("state") != "ready"):
            sys.exit(f"the API must use a ready GPU service (RECEIVER=http); got {st['receiver_health']}")
        seen = misses(c)
        for rec in a.records:
            for scen in SCENARIOS:
                for mode in MODES:
                    body = {"record": rec, "start_s": 0, "duration_s": 300, "mode": mode, "scenario": scen,
                            "max_reviews": 6}
                    d = wait(c, c.post("/v1/runs", json=body).json()["id"])
                    for wid in sorted(d["findings"]):
                        c.post(f"/v1/runs/{d['id']}/windows/{wid}/explain")
                    asked = 0
                    if mode == "balanced" and scen is None and d["findings"]:
                        f = d["findings"]
                        top = sorted(f, key=lambda w: (-["routine", "priority", "urgent"].index(f[w]["final_tier"]), w))[0]
                        unrev = next((w["id"] for w in d["windows"] if w["id"] not in f and w["readable"]), top)
                        for q in QUESTIONS:
                            c.post(f"/v1/runs/{d['id']}/questions",
                                   json={"question": q.format(top=top, tier=f[top]["final_tier"], unreviewed=unrev)})
                            asked += 1
                    srcs = sorted({t["source"] for x in d["findings"].values() for t in x["attempts"]})
                    # Ask the store itself: any request it could not answer (triage, explanation, question routing,
                    # summary) counts, whatever the agents then did with the fallback.
                    now = misses(c)
                    if a.verify and now > seen:
                        missed.append(f"{rec} {mode} {scen or '-'}: {now - seen} request(s) not in the recording")
                    seen = now
                    print(f"{time.time() - t0:6.0f}s {rec} {mode:10s} {scen or '-':11s} {d['status']:10s} "
                          f"windows={len(d['findings'])} sources={srcs} questions={asked}", flush=True)
        print(f"done in {time.time() - t0:.0f} s; replay entries now: "
              f"{c.get('/v1/status').json()['receiver_health'].get('replay_entries')}")
    if a.verify:
        print("\n".join(missed) or "every request was served from the recording (0 misses)")
        sys.exit(1 if missed else 0)


if __name__ == "__main__":
    main()
