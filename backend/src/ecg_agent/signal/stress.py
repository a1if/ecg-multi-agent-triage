"""Signal faults for stress scenarios and tests: what goes wrong with real single-lead recordings.

Each function takes a lead (any sampling rate) and returns a damaged copy. Seeded, so a scenario is reproducible.
Used by the demo's stress-scenario toggles and by the perception agent's quality tests.
"""
from __future__ import annotations

import numpy as np


def noise(x: np.ndarray, snr_db: float, seed: int = 0) -> np.ndarray:
    """White noise at a given signal-to-noise ratio (muscle artefact, poor electrode contact)."""
    p_signal = float(np.mean((x - x.mean()) ** 2))
    sd = np.sqrt(p_signal / 10 ** (snr_db / 10))
    return x + np.random.default_rng(seed).normal(0.0, sd, len(x))


def invert(x: np.ndarray) -> np.ndarray:
    """Lead attached the wrong way round."""
    return -x


def clip(x: np.ndarray, keep: float = 0.3) -> np.ndarray:
    """Amplifier saturation: everything beyond the central ``keep`` fraction of the range is flattened."""
    lo, hi = np.percentile(x, 50 * (1 - keep)), np.percentile(x, 100 - 50 * (1 - keep))
    return np.clip(x, lo, hi)


def dropout(x: np.ndarray, fs: float, fraction: float = 0.5, seed: int = 0) -> np.ndarray:
    """Electrode falling off: flat stretches of 2-10 s covering about ``fraction`` of the recording."""
    rng = np.random.default_rng(seed)
    y, n, covered = x.copy(), len(x), 0
    while covered < fraction * n:
        length = int(rng.uniform(2, 10) * fs)
        start = int(rng.integers(0, max(1, n - length)))
        y[start:start + length] = y[start]
        covered += length
    return y


def wander(x: np.ndarray, fs: float, amplitude: float = 2.0, hz: float = 0.3) -> np.ndarray:
    """Baseline wander (breathing, movement), in multiples of the signal's standard deviation."""
    t = np.arange(len(x)) / fs
    return x + amplitude * float(np.std(x)) * np.sin(2 * np.pi * hz * t)


def noise_burst(x: np.ndarray, fs: float, fraction: float = 0.3, snr_db: float = -6, seed: int = 0) -> np.ndarray:
    """Noise in bursts of 10-30 s covering about ``fraction`` of the recording (patient moving, loose electrode)."""
    rng = np.random.default_rng(seed)
    y, n, covered = x.copy(), len(x), 0
    loud = noise(x, snr_db, seed)
    while covered < fraction * n:
        length = int(rng.uniform(10, 30) * fs)
        start = int(rng.integers(0, max(1, n - length)))
        y[start:start + length] = loud[start:start + length]
        covered += length
    return y


SCENARIOS = {
    "noise_burst": lambda x, fs: noise_burst(x, fs),
    "noise_10db": lambda x, fs: noise(x, 10),
    "noise_0db": lambda x, fs: noise(x, 0),
    "noise_-6db": lambda x, fs: noise(x, -6),
    "inverted": lambda x, fs: invert(x),
    "clipped": lambda x, fs: clip(x, 0.3),
    "dropout_50": lambda x, fs: dropout(x, fs, 0.5),
    "dropout_90": lambda x, fs: dropout(x, fs, 0.9),
    "wander": lambda x, fs: wander(x, fs),
}
