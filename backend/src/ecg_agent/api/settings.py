"""API settings, from environment variables (12-factor: the same image runs locally, in compose and on Cloud Run)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _flag(name: str, default: bool) -> bool:
    return _env(name, "1" if default else "0").lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    records_dir: Path = ROOT / "data" / "records"
    models_dir: Path = ROOT / "artifacts" / "models"
    # Receiver: "offline" (the rule, labelled), "replay" (recorded Gemma answers only) or "http" (the GPU service,
    # with recorded answers first and the offline rule if the service is down).
    receiver: str = "offline"
    inference_url: str = "http://localhost:8001"
    inference_auth: str = "none"  # "none" | "bearer" | "gcp" (ID token from the metadata server)
    inference_token: str | None = None
    replay_store: Path = ROOT / "data" / "replay" / "answers.jsonl"
    planner: str = "rule"  # "rule" | "gemma"
    classifier: str = "rule"  # "rule" | "gemma" (clinician questions)
    allow_uploads: bool = False  # off in the public demo (design §2); on in local / clinician deployments
    max_duration_s: float = 600.0  # longest segment one run may analyse
    max_reviews: int = 6
    max_concurrent_runs: int = 2
    max_stored_runs: int = 50
    runs_per_minute: int = 6  # per client
    questions_per_minute: int = 20  # per client
    cors_origins: tuple[str, ...] = field(default_factory=lambda: ("http://localhost:3000",))
    audit_dir: Path | None = None

    @classmethod
    def from_env(cls) -> Settings:
        d = cls()
        return cls(
            records_dir=Path(_env("RECORDS_DIR", str(d.records_dir))),
            models_dir=Path(_env("MODELS_DIR", str(d.models_dir))),
            receiver=_env("RECEIVER", d.receiver),
            inference_url=_env("INFERENCE_URL", d.inference_url),
            inference_auth=_env("INFERENCE_AUTH", d.inference_auth),
            inference_token=os.environ.get("INFERENCE_TOKEN"),
            replay_store=Path(_env("REPLAY_STORE", str(d.replay_store))),
            planner=_env("PLANNER", d.planner),
            classifier=_env("CLASSIFIER", d.classifier),
            allow_uploads=_flag("ALLOW_UPLOADS", d.allow_uploads),
            max_duration_s=float(_env("MAX_DURATION_S", str(d.max_duration_s))),
            max_reviews=int(_env("MAX_REVIEWS", str(d.max_reviews))),
            max_concurrent_runs=int(_env("MAX_CONCURRENT_RUNS", str(d.max_concurrent_runs))),
            max_stored_runs=int(_env("MAX_STORED_RUNS", str(d.max_stored_runs))),
            runs_per_minute=int(_env("RUNS_PER_MINUTE", str(d.runs_per_minute))),
            questions_per_minute=int(_env("QUESTIONS_PER_MINUTE", str(d.questions_per_minute))),
            cors_origins=tuple(o.strip() for o in _env("CORS_ORIGINS", ",".join(d.cors_origins)).split(",") if o),
            audit_dir=Path(os.environ["AUDIT_DIR"]) if os.environ.get("AUDIT_DIR") else None,
        )
