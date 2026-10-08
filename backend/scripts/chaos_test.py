"""Load and chaos tests against the real services, started as separate processes.

    python scripts/chaos_test.py load     # API (offline receiver): 8 runs at once; all complete, concurrency bounded
    python scripts/chaos_test.py chaos    # API + GPU service: the GPU service is killed mid-run; the run must complete,
                                          # answers before the kill from gemma, after it from the labelled offline rule

GPU_PY / CPU_PY: the python executables of the GPU and CPU environments (default: .venv-gpu and .venv).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
CPU_PY = os.environ.get("CPU_PY", str(ROOT / ".venv/Scripts/python.exe"))
GPU_PY = os.environ.get("GPU_PY", str(ROOT / ".venv-gpu/Scripts/python.exe"))
API, INF = "http://127.0.0.1:8100", "http://127.0.0.1:8101"


def start(py: str, app: str, port: int, env: dict, log: str) -> subprocess.Popen:
    out = open(ROOT / "logs" / log, "w")  # noqa: SIM115 - lives as long as the process
    return subprocess.Popen([py, "-m", "uvicorn", app, "--port", str(port)], cwd=ROOT, stdout=out,
                            stderr=subprocess.STDOUT, env={**os.environ, "PYTHONUNBUFFERED": "1", **env})


def wait_ready(url: str, timeout: float) -> float:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if httpx.get(f"{url}/readyz", timeout=2).status_code == 200:
                return time.time() - t0
        except httpx.HTTPError:
            pass
        time.sleep(1)
    raise TimeoutError(url)


def wait_run(c: httpx.Client, rid: str, timeout: float = 600) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout:
        d = c.get(f"/v1/runs/{rid}").json()
        if d["status"] not in ("queued", "running"):
            return d
        time.sleep(0.5)
    raise TimeoutError(rid)


def load() -> dict:
    api = start(CPU_PY, "ecg_agent.api.main:app", 8100, {"RECEIVER": "offline", "RUNS_PER_MINUTE": "100",
                                                          "MAX_CONCURRENT_RUNS": "2"}, "chaos_api.log")
    try:
        wait_ready(API, 120)
        with httpx.Client(base_url=API, timeout=30) as c:
            records = ["100", "105", "200", "210", "213", "222", "233", "100"]
            t0 = time.time()
            ids = [c.post("/v1/runs", json={"record": r, "duration_s": 600}).json()["id"] for r in records]
            peak, done = 0, {}
            while len(done) < len(ids):
                rows = {r["id"]: r for r in c.get("/v1/runs").json()}
                peak = max(peak, sum(rows[i]["status"] == "running" for i in ids))
                for i in ids:
                    if rows[i]["status"] not in ("queued", "running"):
                        done[i] = rows[i]["status"]
                time.sleep(0.05)
            wall = time.time() - t0
        res = {"runs": len(ids), "statuses": sorted(done.values()), "peak_running": peak, "wall_s": round(wall, 1),
               "ok": all(s == "complete" for s in done.values()) and peak <= 2}
    finally:
        api.terminate()
    return res


def chaos() -> dict:
    (ROOT / "logs/chaos_replay.jsonl").unlink(missing_ok=True)  # no recorded answers: every call is live or fallback
    inf = start(GPU_PY, "ecg_agent.inference.main:app", 8101, {"INFERENCE_TOKEN": "chaos"}, "chaos_inference.log")
    api = start(CPU_PY, "ecg_agent.api.main:app", 8100, {"RECEIVER": "http", "INFERENCE_URL": INF,
                                                          "INFERENCE_TOKEN": "chaos", "MAX_REVIEWS": "10",
                                                          "REPLAY_STORE": str(ROOT / "logs/chaos_replay.jsonl")},
                "chaos_api.log")
    try:
        cold = wait_ready(INF, 300)
        wait_ready(API, 60)
        with httpx.Client(base_url=API, timeout=30) as c:
            rid = c.post("/v1/runs", json={"record": "213", "duration_s": 600, "mode": "accuracy"}).json()["id"]
            killed_at = None
            t0 = time.time()
            while time.time() - t0 < 300:  # kill the GPU service once two windows have been triaged live
                steps = c.get(f"/v1/runs/{rid}").json()
                if steps["status"] not in ("queued", "running"):
                    break
                live = sum(1 for f in steps["findings"].values())
                if live >= 2 and killed_at is None:
                    inf.kill()
                    killed_at = time.time() - t0
                time.sleep(0.2)
            d = wait_run(c, rid)
        tries = [(w, t["channel"], t["source"], t["verdict"]) for w, f in sorted(d["findings"].items())
                 for t in f["attempts"]]
        sources = [s for _, _, s, _ in tries]
        below = [w for w, f in d["findings"].items()
                 if ["routine", "priority", "urgent"].index(f["final_tier"])
                 < ["routine", "priority", "urgent"].index(f["screening_tier"])]
        res = {"inference_cold_start_s": round(cold, 1), "killed_after_s": killed_at, "status": d["status"],
               "attempts": tries, "gemma_answers": sources.count("gemma"),
               "offline_answers": sources.count("offline-rule"), "below_screening": below,
               "narrative_source": d["report"]["narrative_source"],
               "ok": d["status"] == "complete" and killed_at is not None and sources.count("gemma") >= 2
               and sources.count("offline-rule") >= 1 and not below}
    finally:
        inf.kill()
        api.terminate()
    return res


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "load"
    out = {"load": load, "chaos": chaos}[which]()
    (ROOT / "results" / f"chaos_{which}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    sys.exit(0 if out["ok"] else 1)
