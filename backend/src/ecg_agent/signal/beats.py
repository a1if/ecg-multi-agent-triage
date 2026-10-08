"""ECG signal -> beat windows the sender can read.

Two sources:
  * MIT-BIH records with reference annotations: beats at the annotated R-peaks, exactly as the sender was trained
    (research repo data_prep.py), plus the cardiologist label per beat for display and evaluation.
  * Any single-lead signal (CSV upload, other databases): resampled to 360 Hz, R-peaks found by a Pan-Tompkins
    style detector, then the same windowing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import gcd
from pathlib import Path

import numpy as np
from scipy import signal as sps
from scipy.ndimage import maximum_filter1d

FS = 360
HALF = 180  # samples either side of the R-peak -> 360-sample window
RR_GAP_CAP_MS = 11999.0  # the event schema rejects heart rates under 5 bpm

AAMI_MAP = {"N": "N", "L": "N", "R": "N", "e": "N", "j": "N", "A": "S", "a": "S", "J": "S", "S": "S",
            "V": "V", "E": "V", "F": "F", "P": "Q", "/": "Q", "f": "Q", "u": "Q"}


@dataclass
class Beats:
    signal: np.ndarray  # the full lead at 360 Hz
    peaks: np.ndarray  # (n,) R-peak sample indices
    windows: np.ndarray  # (n, 360)
    rr_ms: np.ndarray  # (n,) pre-RR in ms
    reference: list[str] | None = None  # annotated AAMI class per beat, when known
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.peaks)


def resample(x: np.ndarray, fs: float) -> np.ndarray:
    if fs == FS:
        return np.asarray(x, dtype=np.float64)
    up, down = FS, int(round(fs))
    g = gcd(up, down)
    return sps.resample_poly(np.asarray(x, dtype=np.float64), up // g, down // g)


def detect_r_peaks(x: np.ndarray, fs: int = FS) -> np.ndarray:
    """Pan-Tompkins style: 5-15 Hz band-pass, derivative, square, 150 ms integration, adaptive peak picking,
    then each peak is moved to the largest |deflection| of the filtered lead within 75 ms."""
    if len(x) < fs:
        return np.zeros(0, dtype=int)
    b, a = sps.butter(2, [5 / (fs / 2), 15 / (fs / 2)], btype="band")
    bp = sps.filtfilt(b, a, x)
    integ = np.convolve(np.diff(bp, prepend=bp[0]) ** 2, np.ones(int(0.15 * fs)) / (0.15 * fs), mode="same")
    # Threshold against the local maximum over 2.5 s, so amplitude drift in a long recording is tolerated.
    thr = np.maximum(0.3 * maximum_filter1d(integ, int(2.5 * fs)), 0.05 * np.percentile(integ, 99))
    allc, _ = sps.find_peaks(integ, distance=int(0.25 * fs))
    cand = allc[integ[allc] > thr[allc]]
    # Searchback: where a gap exceeds 1.66x the local RR, accept the strongest candidate above half the threshold.
    if len(cand) > 2:
        extra, overall = [], float(np.median(np.diff(cand)))
        for k, (a, b) in enumerate(zip(cand[:-1], cand[1:], strict=True)):
            local = float(np.median(np.diff(cand[max(0, k - 8):k + 1]))) if k >= 1 else overall
            if b - a > 1.66 * local:
                inside = allc[(allc > a + 0.2 * fs) & (allc < b - 0.2 * fs)]
                inside = inside[integ[inside] > 0.5 * thr[inside]]
                if len(inside):
                    extra.append(inside[np.argmax(integ[inside])])
        cand = np.sort(np.concatenate([cand, np.asarray(extra, dtype=int)]))
    r = int(0.075 * fs)
    peaks = np.unique(np.array([max(0, c - r) + int(np.argmax(np.abs(bp[max(0, c - r):c + r]))) for c in cand],
                               dtype=int))
    # Refinement can pull two candidates within one refractory period (seen on clipped signals: 166 ms apart, an
    # impossible 360 bpm). No two beats closer than 200 ms: keep the stronger.
    kept: list[int] = []
    for p in peaks:
        if kept and p - kept[-1] < int(0.2 * fs):
            if abs(bp[p]) > abs(bp[kept[-1]]):
                kept[-1] = int(p)
        else:
            kept.append(int(p))
    return np.asarray(kept, dtype=int)


def windows_at(x: np.ndarray, peaks: np.ndarray):
    keep = (peaks - HALF >= 0) & (peaks + HALF <= len(x))
    peaks = peaks[keep]
    w = np.stack([x[p - HALF:p + HALF] for p in peaks]).astype(np.float32) if len(peaks) else np.zeros((0, 2 * HALF),
                                                                                                         np.float32)
    return peaks, keep, w


def rr_from_peaks(peaks: np.ndarray) -> np.ndarray:
    rr = np.diff(peaks).astype(np.float64) / FS * 1000.0
    first = float(np.median(rr)) if len(rr) else 800.0
    return np.minimum(np.concatenate([[first], rr]), RR_GAP_CAP_MS)


def from_signal(x: np.ndarray, fs: float, meta: dict | None = None) -> Beats:
    x = resample(np.asarray(x, dtype=np.float64).ravel(), fs)
    x = x - sps.medfilt(x, 2 * (FS // 5) + 1) if len(x) > FS else x  # remove baseline wander
    peaks, _, w = windows_at(x, detect_r_peaks(x))
    return Beats(x, peaks, w, rr_from_peaks(peaks), None, {"source": "detector", **(meta or {})})


def from_mitdb(record_path: str | Path, start_s: float = 0.0, duration_s: float | None = None) -> Beats:
    """Beats at the reference annotations, the sender's training setting. RR is measured to the previous accepted
    beat, as in training."""
    import wfdb

    rec = wfdb.rdrecord(str(record_path))
    ann = wfdb.rdann(str(record_path), "atr")
    names = rec.sig_name
    lead = rec.p_signal[:, names.index("MLII") if "MLII" in names else 0].astype(np.float64)
    if rec.fs != FS:
        raise ValueError(f"MIT-BIH loader expects {FS} Hz, got {rec.fs}")
    lo = int(start_s * FS)
    hi = len(lead) if duration_s is None else min(len(lead), lo + int(duration_s * FS))

    peaks, labels = [], []
    for s, sym in zip(ann.sample, ann.symbol, strict=True):
        if sym in AAMI_MAP and s - HALF >= 0 and s + HALF <= len(lead):
            peaks.append(int(s))
            labels.append(AAMI_MAP[sym])
    peaks = np.asarray(peaks)
    rr = rr_from_peaks(peaks)  # over the whole record, so the segment's first beat has its true RR
    sel = (peaks >= lo) & (peaks < hi)
    peaks, rr = peaks[sel], rr[sel]
    labels = [l for l, k in zip(labels, sel, strict=True) if k]
    w = np.stack([lead[p - HALF:p + HALF] for p in peaks]).astype(np.float32) if len(peaks) else np.zeros((0, 360))
    seg = lead[lo:hi]
    return Beats(seg, peaks - lo, w, rr, labels,
                 {"source": "annotations", "record": Path(record_path).name, "start_s": start_s,
                  "duration_s": (hi - lo) / FS, "full_duration_s": len(lead) / FS})


def match_peaks(detected: np.ndarray, reference: np.ndarray, tol: int = int(0.15 * FS)) -> dict:
    """Beat-detection sensitivity / positive predictivity with the standard 150 ms matching window."""
    reference, detected = np.sort(reference), np.sort(detected)
    if len(reference) == 0 or len(detected) == 0:
        return {"tp": 0, "fn": len(reference), "fp": len(detected), "se": 0.0, "ppv": 0.0}
    idx = np.clip(np.searchsorted(detected, reference), 1, len(detected) - 1)
    nearest = np.minimum(np.abs(detected[idx] - reference), np.abs(detected[idx - 1] - reference))
    tp = int((nearest <= tol).sum())
    fn, fp = len(reference) - tp, max(0, len(detected) - tp)
    return {"tp": tp, "fn": fn, "fp": fp, "se": tp / len(reference), "ppv": tp / max(1, tp + fp)}
