"""The frontend's only door to the backend: every HTTP call to the API service lives here.

The frontend holds no logic or safety rules of its own. It shows what the API returns and sends what the user asks,
so it can be replaced (e.g. by a React app) without touching the agents.
"""
from __future__ import annotations

import os

import httpx

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000").rstrip("/")
# On Cloud Run the API accepts calls only from this frontend's service account: each call carries a Google-signed
# ID token from the metadata server (no key or shared secret anywhere). Locally API_AUTH is unset: no token.
API_AUTH = os.environ.get("API_AUTH", "none")
_client = httpx.Client(base_url=API_URL, timeout=30.0)
_token: dict = {"value": None, "expires": 0.0}


def _auth_headers() -> dict:
    if API_AUTH != "gcp":
        return {}
    import time

    if _token["value"] is None or time.time() > _token["expires"]:
        r = httpx.get("http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/identity",
                      params={"audience": API_URL}, headers={"Metadata-Flavor": "Google"}, timeout=5.0)
        r.raise_for_status()
        _token["value"], _token["expires"] = r.text, time.time() + 50 * 60  # tokens live 60 min
    return {"Authorization": f"Bearer {_token['value']}"}


class ApiError(RuntimeError):
    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status, self.detail = status, detail


def _call(method: str, path: str, **kw):
    try:
        r = _client.request(method, path, headers=_auth_headers(), **kw)
    except httpx.HTTPError as exc:
        raise ApiError(0, f"cannot reach the API at {API_URL} ({type(exc).__name__})") from exc
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        raise ApiError(r.status_code, str(detail))
    return r.json()


def status() -> dict:
    return _call("GET", "/v1/status")


def records() -> list[dict]:
    return _call("GET", "/v1/records")


def scenarios() -> list[str]:
    return _call("GET", "/v1/scenarios")


def worklist() -> list[dict]:
    return _call("GET", "/v1/runs")


def create_run(record: str, start_s: float, duration_s: float, mode: str, scenario: str | None,
               max_reviews: int) -> dict:
    return _call("POST", "/v1/runs", json={"record": record, "start_s": start_s, "duration_s": duration_s,
                                           "mode": mode, "scenario": scenario, "max_reviews": max_reviews})


def upload_run(name: str, data: bytes, fs: float, mode: str) -> dict:
    return _call("POST", "/v1/runs/upload", files={"file": (name, data)}, data={"fs": str(fs), "mode": mode})


def run(run_id: str) -> dict:
    return _call("GET", f"/v1/runs/{run_id}")


def steps(run_id: str, after: int = -1) -> dict:
    return _call("GET", f"/v1/runs/{run_id}/steps", params={"after": after})


def signal(run_id: str, start_s: float, end_s: float, max_points: int = 3000) -> dict:
    return _call("GET", f"/v1/runs/{run_id}/signal", params={"start_s": start_s, "end_s": end_s,
                                                              "max_points": max_points})


def audit(run_id: str) -> list[dict]:
    return _call("GET", f"/v1/runs/{run_id}/audit")


def ask(run_id: str, question: str) -> dict:
    return _call("POST", f"/v1/runs/{run_id}/questions", json={"question": question})


def explain(run_id: str, wid: str) -> dict:
    return _call("POST", f"/v1/runs/{run_id}/windows/{wid}/explain")


def override(run_id: str, wid: str, tier: str, clinician: str, reason: str) -> dict:
    return _call("POST", f"/v1/runs/{run_id}/windows/{wid}/override",
                 json={"tier": tier, "clinician": clinician, "reason": reason})
