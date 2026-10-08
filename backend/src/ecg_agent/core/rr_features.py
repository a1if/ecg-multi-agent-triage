"""RR-interval features for the RR-branch encoder (Phase 1, encoder sweep).

Supraventricular ectopic (S) beats usually look like normal beats. What sets them
apart is timing: a short pre-RR, a longer post-RR, relative to the patient's own
rhythm. The reference CNN-LSTM sees one 1-s window and no RR input, which is the
main reason its DS2 S recall is 8.2%. These are the de Chazal-style interval
features, normalised against the patient's local rhythm so they carry across
patients (inter-patient split).

Features per beat, computed within each record only (never across records):
    pre_s       pre-RR (s)
    post_s      post-RR (s): the next beat's pre-RR. Needs the next beat, so an
                online agent emits each beat one beat late. The last beat in a
                record falls back to its pre-RR.
    local_s     mean pre-RR of the previous LOCAL_WINDOW beats (causal, excludes
                the current beat); falls back to pre-RR for a record's first beat
    pre_ratio   pre_s / local_s
    post_ratio  post_s / local_s

RR values are clipped to [RR_MIN_MS, RR_MAX_MS] first: DS1 contains gaps up to
~100 s where annotations or edge windows were skipped (data_prep.py).
"""
import numpy as np

RR_MIN_MS = 200.0
RR_MAX_MS = 3000.0
LOCAL_WINDOW = 10
FEATURE_NAMES = ("pre_s", "post_s", "local_s", "pre_ratio", "post_ratio")
N_RR_FEATURES = len(FEATURE_NAMES)


def clip_rr_s(rr_interval_ms: float) -> float:
    return min(max(float(rr_interval_ms), RR_MIN_MS), RR_MAX_MS) / 1000.0


def beat_rr_features(pre_s: float, post_s: float | None, previous_pre_s) -> np.ndarray:
    """Features for one beat. ``pre_s``/``post_s`` are clipped RRs in seconds
    (``post_s`` None -> the beat's own pre-RR, as for a record's last beat);
    ``previous_pre_s`` holds the clipped pre-RRs of earlier beats in the same
    record, most recent last (only the last LOCAL_WINDOW are used). Shared by
    the batch path below and by PerceptionAgent, so training and inference
    features cannot diverge."""
    post_s = pre_s if post_s is None else post_s
    window = list(previous_pre_s)[-LOCAL_WINDOW:]
    local_s = float(np.mean(window)) if window else pre_s
    return np.array([pre_s, post_s, local_s, pre_s / local_s, post_s / local_s], dtype=np.float32)


def compute_rr_features(rr_interval_ms: np.ndarray, record_ids: np.ndarray) -> np.ndarray:
    """(n,) pre-RR in ms + (n,) record ids, in chronological order within each
    record -> (n, 5) float32 features. Records must be contiguous."""
    record_ids = np.asarray(record_ids)
    boundaries = np.flatnonzero(np.diff(record_ids) != 0) + 1
    if len(np.unique(record_ids)) != len(boundaries) + 1:
        raise ValueError("record_ids must be contiguous (one run per record)")

    out = np.empty((len(record_ids), N_RR_FEATURES), dtype=np.float32)
    for seg in np.split(np.arange(len(record_ids)), boundaries):
        pre = [clip_rr_s(v) for v in np.asarray(rr_interval_ms)[seg]]
        for j, i in enumerate(seg):
            post = pre[j + 1] if j + 1 < len(pre) else None
            out[i] = beat_rr_features(pre[j], post, pre[max(0, j - LOCAL_WINDOW):j])
    return out


def fit_standardizer(features: np.ndarray) -> dict:
    """Per-feature mean/std from training data only; saved with the checkpoint."""
    return {"mean": features.mean(axis=0).tolist(), "std": (features.std(axis=0) + 1e-6).tolist()}


def standardize(features: np.ndarray, stats: dict) -> np.ndarray:
    return ((features - np.asarray(stats["mean"])) / np.asarray(stats["std"])).astype(np.float32)
