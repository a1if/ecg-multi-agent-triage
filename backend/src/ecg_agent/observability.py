"""Prometheus metrics and JSON logging shared by both services (Cloud Logging parses the JSON lines)."""
from __future__ import annotations

import json
import logging
import sys
import time
from types import SimpleNamespace

from prometheus_client import Counter, Gauge, Histogram

metrics = SimpleNamespace(
    runs=Counter("agent_runs_total", "Agent runs by outcome", ["status"]),
    run_seconds=Histogram("agent_run_seconds", "Wall time of a full agent run",
                          buckets=(1, 2, 5, 10, 20, 40, 80, 160, 320, 640)),
    messages=Counter("agent_messages_total", "Inter-agent messages", ["sender", "intent"]),
    tool_calls=Counter("agent_tool_calls_total", "Tool calls", ["agent", "tool", "status"]),
    tool_latency=Histogram("agent_tool_seconds", "Tool call latency", ["agent", "tool"],
                           buckets=(0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 30, 60, 300)),
    guardrail=Counter("agent_guardrail_total", "Guardrail verdicts on receiver answers", ["channel", "verdict"]),
    channel=Counter("agent_channel_total", "Windows sent per channel", ["channel"]),
    receiver_tokens=Histogram("receiver_prompt_tokens", "Receiver prompt tokens", ["channel"],
                              buckets=(256, 512, 768, 1024, 1536, 2048, 3072, 4096)),
    receiver_latency=Histogram("receiver_latency_seconds", "Receiver decision latency", ["channel", "source"],
                               buckets=(0.05, 0.1, 0.5, 1, 2, 4, 8, 16, 32, 64)),
    # Input (data) drift: what the perception agent is being given, recording by recording.
    input_noise=Histogram("perception_input_noise", "Median share of beat energy above 40 Hz, per recording",
                          buckets=(0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.15, 0.3, 1.0)),
    windows=Counter("perception_windows_total", "Screened windows by readability", ["readable"]),
    beats=Counter("perception_beats_total", "Beats by predicted class", ["label"]),
    receiver_mode=Gauge("agent_receiver_mode", "Receiver this deployment is configured to use (1 = active)", ["mode"]),
    gpu_mem=Gauge("inference_gpu_memory_bytes", "Peak GPU memory of the last request"),
    inflight=Gauge("inference_inflight", "Requests waiting for or using the GPU"),
)


class _JsonFormatter(logging.Formatter):
    def format(self, r: logging.LogRecord) -> str:
        out = {"severity": r.levelname, "message": r.getMessage(), "logger": r.name,
               "time": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(r.created))}
        out.update(getattr(r, "extra_fields", {}))
        if r.exc_info:
            out["exception"] = self.formatException(r.exc_info)
        return json.dumps(out)


def setup_logging(level: str = "INFO") -> None:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(_JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [h]
    root.setLevel(level)
