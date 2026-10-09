"""Start the services locally with one command, wait until they are ready, stop them with Ctrl+C.

    python scripts/serve_local.py              # API only, offline receiver (no GPU): answers come from the rule
    python scripts/serve_local.py --gpu        # GPU inference service + API using it (recorded answers first)
    python scripts/serve_local.py --gpu --gemma-everywhere   # Gemma also plans and routes questions
    python scripts/serve_local.py --uploads    # allow CSV uploads (local use only)
    python scripts/serve_local.py --no-ui      # services only, no Streamlit frontend

Then open http://127.0.0.1:8501 (frontend) or http://127.0.0.1:8000/docs (interactive API).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
CPU_PY = os.environ.get("CPU_PY", str(ROOT / ".venv/Scripts/python.exe"))
GPU_PY = os.environ.get("GPU_PY", str(ROOT / ".venv-gpu/Scripts/python.exe"))


def ready(url: str, timeout: float, proc: subprocess.Popen, name: str) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            sys.exit(f"{name} exited early; see logs/{name}.log")
        try:
            if httpx.get(f"{url}/readyz", timeout=2).status_code == 200:
                print(f"  {name} ready in {time.time() - t0:.0f} s at {url}")
                return
        except httpx.HTTPError:
            pass
        time.sleep(1)
    sys.exit(f"{name} not ready after {timeout:.0f} s; see logs/{name}.log")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true", help="also start the GPU inference service and use it")
    ap.add_argument("--gemma-everywhere", action="store_true", help="Gemma plans and routes questions too")
    ap.add_argument("--uploads", action="store_true", help="allow CSV uploads")
    ap.add_argument("--no-ui", action="store_true", help="do not start the Streamlit frontend")
    a = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # print progress at once, even when redirected to a file
    (ROOT / "logs").mkdir(exist_ok=True)
    procs, token = [], "local-dev"
    base = {**os.environ, "PYTHONUNBUFFERED": "1"}
    try:
        if a.gpu:
            print("Starting the GPU inference service (loads Gemma: about a minute)...")
            inf = subprocess.Popen([GPU_PY, "-m", "uvicorn", "ecg_agent.inference.main:app", "--port", "8001"],
                                   cwd=ROOT, env={**base, "INFERENCE_TOKEN": token},
                                   stdout=open(ROOT / "logs/inference.log", "w"), stderr=subprocess.STDOUT)
            procs.append(inf)
            ready("http://127.0.0.1:8001", 300, inf, "inference")
        env = {**base, "RECEIVER": "http" if a.gpu else "offline", "INFERENCE_URL": "http://127.0.0.1:8001",
               "INFERENCE_TOKEN": token, "ALLOW_UPLOADS": "1" if a.uploads else "0",
               "PLANNER": "gemma" if a.gemma_everywhere else "rule",
               "CLASSIFIER": "gemma" if a.gpu else "rule", "AUDIT_DIR": str(ROOT / "logs/audit")}
        print(f"Starting the API (receiver: {env['RECEIVER']})...")
        api = subprocess.Popen([CPU_PY, "-m", "uvicorn", "ecg_agent.api.main:app", "--port", "8000"], cwd=ROOT,
                               env=env, stdout=open(ROOT / "logs/api.log", "w"), stderr=subprocess.STDOUT)
        procs.append(api)
        ready("http://127.0.0.1:8000", 120, api, "api")
        if not a.no_ui:
            print("Starting the frontend...")
            ui = subprocess.Popen([CPU_PY, "-m", "streamlit", "run", "app.py", "--server.port", "8501",
                                   "--server.address", "127.0.0.1", "--server.headless", "true",
                                   "--browser.gatherUsageStats", "false"], cwd=ROOT.parent / "frontend",
                                  env={**base, "API_URL": "http://127.0.0.1:8000"},
                                  stdout=open(ROOT / "logs/frontend.log", "w"), stderr=subprocess.STDOUT)
            procs.append(ui)
            t0 = time.time()
            while time.time() - t0 < 60:
                try:
                    if httpx.get("http://127.0.0.1:8501/_stcore/health", timeout=2).status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(1)
        print("\nReady.")
        if not a.no_ui:
            print("  Frontend:        http://127.0.0.1:8501")
        print("  Interactive API: http://127.0.0.1:8000/docs\n(Ctrl+C to stop)")
        while all(p.poll() is None for p in procs):
            time.sleep(1)
        print("A service stopped; see logs/.")
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        for p in procs:
            p.terminate()


if __name__ == "__main__":
    main()
