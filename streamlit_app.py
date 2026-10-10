"""Streamlit Community Cloud entry point (free hosting, no Docker).

Community Cloud runs one Streamlit app, so the API runs inside this process: a uvicorn server in a background thread,
listening on 127.0.0.1 only (not reachable from outside, like the private API on Cloud Run or in the Space container).
The frontend is the same code as frontend/app.py. The reasoning agent serves the Gemma answers recorded on a GPU for
the demo presets (backend/data/replay/answers.jsonl), labelled as recorded; nothing here needs a GPU.

Everywhere else (local, Docker, CI, GCP) the API and frontend are separate services; see docker-compose.yml.
"""
import os
import runpy
import sys
import threading
import time
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "frontend"))  # the frontend's own modules (api, ui)

for key, value in {
    "RECEIVER": "replay", "DEMO_PRESETS": "1", "CLASSIFIER": "gemma", "ALLOW_UPLOADS": "0",
    "RUNS_PER_MINUTE": "6", "QUESTIONS_PER_MINUTE": "30", "MAX_STORED_RUNS": "30",
    "RECORDS_DIR": str(ROOT / "backend/data/records"), "MODELS_DIR": str(ROOT / "backend/artifacts/models"),
    "REPLAY_STORE": str(ROOT / "backend/data/replay/answers.jsonl"), "API_URL": "http://127.0.0.1:8000",
}.items():
    os.environ.setdefault(key, value)


@st.cache_resource(show_spinner="Starting the two agents (first visit after a pause takes a few seconds)...")
def start_api():
    """Once per process: start the API in a background thread and wait until it is ready."""
    import httpx
    import uvicorn

    from ecg_agent.api.main import create_app
    from ecg_agent.api.settings import Settings

    server = uvicorn.Server(uvicorn.Config(create_app(Settings.from_env()), host="127.0.0.1", port=8000,
                                           log_level="warning"))
    threading.Thread(target=server.run, name="ecg-api", daemon=True).start()
    for _ in range(240):
        try:
            if httpx.get("http://127.0.0.1:8000/readyz", timeout=2).status_code == 200:
                return server
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    raise RuntimeError("the embedded API did not start within 2 minutes")


start_api()
runpy.run_path(str(ROOT / "frontend" / "app.py"), run_name="__main__")
