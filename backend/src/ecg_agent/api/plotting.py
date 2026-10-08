"""Signal for the browser: never the raw samples of a long recording, a min-max envelope sized to the screen.

Min-max decimation keeps each bin's lowest and highest sample in time order, so narrow QRS spikes survive (plain
averaging or striding would erase them). Zoomed in far enough, the raw samples are returned unchanged.
"""
from __future__ import annotations

import numpy as np

FS = 360


def envelope(x: np.ndarray, start: int, end: int, max_points: int) -> tuple[list[float], list[float]]:
    """-> (times in s, values) for samples [start, end) with at most ``max_points`` points."""
    seg = np.asarray(x[start:end], dtype=np.float64)
    n = len(seg)
    if n <= max_points:
        return [round((start + i) / FS, 4) for i in range(n)], [round(float(v), 4) for v in seg]
    bins = max(1, max_points // 2)
    edges = np.linspace(0, n, bins + 1).astype(int)
    ts, vs = [], []
    for a, b in zip(edges[:-1], edges[1:], strict=True):
        if b <= a:
            continue
        chunk = seg[a:b]
        i, j = int(np.argmin(chunk)), int(np.argmax(chunk))
        for k in sorted((i, j)):
            ts.append(round((start + a + k) / FS, 4))
            vs.append(round(float(chunk[k]), 4))
    return ts, vs


def signal_view(perception, start_s: float, end_s: float, max_points: int = 2000) -> dict:
    """The signal envelope plus every beat in the range: time, class, confidence, tier, reference label."""
    if not perception.best or perception.unreadable:
        return {"t": [], "v": [], "beats": [], "fs": FS}
    b, out = perception.best.beats, perception.best.out
    sig = b.signal
    start, end = max(0, int(start_s * FS)), min(len(sig), int(end_s * FS))
    t, v = envelope(sig, start, end, max(100, min(max_points, 20000)))
    idx = np.flatnonzero((b.peaks >= start) & (b.peaks < end))
    win_of = {}
    for w in perception.windows.values():
        for i in range(w.start, w.start + w.n):
            win_of[i] = w.id
    from ecg_agent.core.rule import beat_tier

    beats = [{"i": int(i), "t": round(float(b.peaks[i]) / FS, 3), "label": out.events[i]["classification"]["label"],
              "confidence": out.events[i]["classification"]["confidence"], "tier": beat_tier(out.events[i]),
              "reference": b.reference[i] if b.reference else None, "window": win_of.get(int(i))} for i in idx]
    return {"t": t, "v": v, "beats": beats, "fs": FS, "duration_s": round(len(sig) / FS, 2)}
