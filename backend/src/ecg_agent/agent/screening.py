"""Screening on the sender's side: fixed windows over the record, ranked by the triage rule, and a record summary."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ecg_agent.core.rule import RANK, beat_tier, window_tier


@dataclass
class Window:
    id: str
    start: int
    n: int
    abnormal: int
    classes: dict[str, int]
    rule_tier: str
    t_start_s: float
    t_end_s: float
    noise: float = 0.0  # median share of beat energy above 40 Hz
    readable: bool = True  # False: too noisy to classify reliably; manual review, never sent for triage

    def public(self) -> dict:
        return self.__dict__.copy()


def make_windows(events: list[dict], peaks: np.ndarray, size: int = 50, fs: int = 360) -> list[Window]:
    """Contiguous windows of ``size`` beats (the adapter's maximum), the last one shorter."""
    out = []
    for k, s in enumerate(range(0, len(events), size)):
        ev = events[s:s + size]
        classes: dict[str, int] = {}
        for e in ev:
            classes[e["classification"]["label"]] = classes.get(e["classification"]["label"], 0) + 1
        out.append(Window(f"w{k:03d}", s, len(ev), sum(e["classification"]["label"] != "N" for e in ev), classes,
                          window_tier(ev), round(float(peaks[s]) / fs, 2), round(float(peaks[s + len(ev) - 1]) / fs, 2)))
    return out


def rank_windows(windows: list[Window]) -> list[Window]:
    """Readable windows first; among them the most urgent by the rule, then the most abnormal beats."""
    return sorted(windows, key=lambda w: (not w.readable, -RANK[w.rule_tier], -w.abnormal, w.start))


def record_summary(events: list[dict], duration_s: float) -> dict:
    hr = np.array([e["signal_features"]["heart_rate_bpm"] for e in events]) if events else np.zeros(1)
    classes: dict[str, int] = {}
    tiers = {"routine": 0, "priority": 0, "urgent": 0}
    longest = 0
    for e in events:
        classes[e["classification"]["label"]] = classes.get(e["classification"]["label"], 0) + 1
        tiers[beat_tier(e)] += 1
        longest = max(longest, e["clinical_flags"]["consecutive_abnormal_beats"])
    return {"beats": len(events), "duration_s": round(duration_s, 1), "hr_median": round(float(np.median(hr)), 1),
            "hr_p05": round(float(np.percentile(hr, 5)), 1), "hr_p95": round(float(np.percentile(hr, 95)), 1),
            "classes": classes, "beat_tiers": tiers, "longest_abnormal_run": longest}
