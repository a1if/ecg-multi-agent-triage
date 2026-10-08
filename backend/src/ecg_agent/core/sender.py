"""The sender: beat windows of one recording -> validated HealthEvent dicts + 32-d context vectors.

Same decisions as the research PerceptionAgent (perception/perception_agent.py), batched over a whole record:
the network runs on every beat at once, then the per-record escalation state (consecutive abnormal beats) is
applied in order. RR features come from ``compute_rr_features``, which builds exactly the per-beat features the
sequential agent builds (pre-RR, next beat's pre-RR, causal local mean), so the outputs match beat for beat.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ecg_agent.core.models import AAMI_CLASSES, CNNLSTMRR
from ecg_agent.core.rr_features import compute_rr_features, standardize
from ecg_agent.core.schema import HealthEventJSON

WINDOW_LEN = 360
SAMPLE_RATE_HZ = 360
QRS_WIDE_THRESHOLD_MS = 120.0
SQI_LOW_THRESHOLD = 0.5
URGENT_CONFIDENCE = 0.85
URGENT_RUN = 3
DESCRIPTIONS = {"N": "Normal sinus beat", "S": "Supraventricular ectopic beat", "V": "Ventricular ectopic beat",
                "F": "Fusion beat", "Q": "Unclassifiable beat"}


def compute_sqi(window: np.ndarray) -> float:
    """Flatline/clipping heuristic in [0, 1]; 1.0 = clean."""
    if window.std() < 1e-6:
        return 0.0
    diffs = np.abs(np.diff(window))
    flat = float((diffs < 1e-4 * (window.max() - window.min() + 1e-8)).mean())
    lo, hi = window.min(), window.max()
    clip = float(((window <= lo + 1e-8) | (window >= hi - 1e-8)).mean())
    return max(0.0, 1.0 - min(1.0, flat + clip))


def estimate_qrs_ms(window: np.ndarray) -> float:
    center = len(window) // 2
    a = np.abs(window - np.median(window))
    thr = 0.25 * a.max() if a.max() > 0 else 0.0
    left = center
    while left > 0 and a[left] > thr:
        left -= 1
    right = center
    while right < len(window) - 1 and a[right] > thr:
        right += 1
    return max(1, right - left) / SAMPLE_RATE_HZ * 1000.0


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass
class SenderOutput:
    events: list[dict]
    vectors: np.ndarray  # (n, 32) float32
    probs: np.ndarray  # (n, 5)


class Sender:
    def __init__(self, checkpoint: str | Path, device: str | None = None, batch_size: int = 512):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ck = torch.load(checkpoint, map_location=self.device, weights_only=False)
        if not (isinstance(ck, dict) and "rr_standardizer" in ck):
            raise ValueError(f"{checkpoint} is not an RR-branch sender checkpoint")
        self.model = CNNLSTMRR().to(self.device)
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval()
        self.rr_stats = ck["rr_standardizer"]
        self.batch_size = batch_size
        self.checkpoint_sha256 = file_sha256(checkpoint)

    @torch.no_grad()
    def process(self, windows: np.ndarray, rr_ms: np.ndarray, record_id: int = 0) -> SenderOutput:
        """windows (n, 360) raw beat windows of ONE record in time order; rr_ms (n,) pre-RR intervals."""
        windows = np.asarray(windows, dtype=np.float32)
        n = len(windows)
        if windows.ndim != 2 or windows.shape[1] != WINDOW_LEN:
            raise ValueError(f"expected (n, {WINDOW_LEN}) windows, got {windows.shape}")
        rr_ms = np.asarray(rr_ms, dtype=np.float64)
        sqi = np.array([compute_sqi(w) for w in windows])
        mean, std = windows.mean(axis=1, keepdims=True), windows.std(axis=1, keepdims=True)
        norm = np.where(std > 1e-8, (windows - mean) / np.where(std > 1e-8, std, 1.0), windows)
        rr_feat = standardize(compute_rr_features(rr_ms, np.full(n, record_id)), self.rr_stats)

        probs, vecs = [], []
        for s in range(0, n, self.batch_size):
            x = torch.from_numpy(norm[s:s + self.batch_size]).float().unsqueeze(1).to(self.device)
            r = torch.from_numpy(rr_feat[s:s + self.batch_size]).to(self.device)
            logits, ctx = self.model(x, r)
            probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
            vecs.append(ctx.cpu().numpy())
        probs = np.concatenate(probs) if n else np.zeros((0, 5))
        vecs = (np.concatenate(vecs) if n else np.zeros((0, 32))).astype(np.float32)

        events, run = [], 0
        for i in range(n):
            label, conf, top3, urgent, reason, run = self._decide(probs[i], sqi[i], run)
            rr = float(rr_ms[i])
            qrs = estimate_qrs_ms(windows[i])
            morph = "wide_complex" if qrs >= QRS_WIDE_THRESHOLD_MS else "narrow_complex"
            if label == "N" and morph == "narrow_complex":
                morph = "normal"
            event = {
                "event_id": f"evt_{record_id}_{i:04d}",
                "timestamp": f"beat-{i}",
                "classification": {"label": label, "description": DESCRIPTIONS[label],
                                   "confidence": round(conf, 6), "top_3": top3},
                "signal_features": {"rr_interval_ms": round(rr, 2), "qrs_duration_ms": round(qrs, 2),
                                    "heart_rate_bpm": round(60000.0 / rr, 2), "beat_morphology": morph},
                "segment_metadata": {"window_samples": WINDOW_LEN, "sample_rate_hz": SAMPLE_RATE_HZ,
                                     "lead": "MLII", "signal_quality_index": round(float(sqi[i]), 4)},
                "clinical_flags": {"requires_urgent_review": bool(urgent), "flag_reason": reason,
                                   "consecutive_abnormal_beats": run},
            }
            HealthEventJSON(**event)  # fail here, not downstream in the receiver
            events.append(event)
        return SenderOutput(events, vecs, probs)

    @staticmethod
    def _decide(p: np.ndarray, sqi: float, run: int):
        if sqi < SQI_LOW_THRESHOLD:
            c = 1.0 - sqi
            top3 = [{"label": "Q", "confidence": round(c, 6)}, {"label": "N", "confidence": round((1 - c) * 0.6, 6)},
                    {"label": "S", "confidence": round((1 - c) * 0.4, 6)}]
            return "Q", c, top3, True, "low_signal_quality", 0
        top = np.argsort(p)[::-1][:3]
        raw = p[top].astype(np.float64)
        renorm = raw / raw.sum()
        label, conf = AAMI_CLASSES[top[0]], float(raw[0])  # true softmax, not the renormalised top-3
        top3 = [{"label": AAMI_CLASSES[top[k]], "confidence": round(float(renorm[k]), 6)} for k in range(3)]
        if label == "N":
            return label, conf, top3, False, None, 0
        run += 1
        urgent = (label in ("V", "F") and conf > URGENT_CONFIDENCE) or run >= URGENT_RUN
        reason = None
        if urgent:
            reason = (f"{run} consecutive abnormal beats ({label}) detected" if run >= URGENT_RUN
                      else f"{label}-class beat detected with confidence {conf:.2f}")
        return label, conf, top3, urgent, reason, run
