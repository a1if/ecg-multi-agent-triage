"""
HealthEventJSON v1.0 — the sender's per-beat event schema (vendored from the research repo).
"""
import re
from typing import Literal

from pydantic import BaseModel, field_validator, model_validator

AAMI_LABELS = Literal["N", "S", "V", "F", "Q"]


class LabelProb(BaseModel):
    label: AAMI_LABELS
    confidence: float

    @field_validator("confidence")
    @classmethod
    def confidence_in_range(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError("confidence must be in [0.0, 1.0]")
        return v


class ClassificationResult(BaseModel):
    label: AAMI_LABELS
    description: str
    confidence: float
    top_3: list[LabelProb]

    @field_validator("description")
    @classmethod
    def description_max_len(cls, v):
        if len(v) > 60:
            raise ValueError("description must be <= 60 chars")
        return v

    @field_validator("confidence")
    @classmethod
    def confidence_in_range(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError("confidence must be in [0.0, 1.0]")
        return v

    @field_validator("top_3")
    @classmethod
    def top_3_length_and_order(cls, v):
        if len(v) != 3:
            raise ValueError("top_3 must have exactly 3 entries")
        confidences = [p.confidence for p in v]
        if confidences != sorted(confidences, reverse=True):
            raise ValueError("top_3 must be sorted descending by confidence")
        return v

    @model_validator(mode="after")
    def top_3_sums_to_one(self):
        total = sum(p.confidence for p in self.top_3)
        if abs(total - 1.0) > 1e-5:
            raise ValueError(f"top_3 confidences must sum to 1.0 (+/- 1e-5), got {total}")
        return self


class SignalFeatures(BaseModel):
    rr_interval_ms: float
    qrs_duration_ms: float
    heart_rate_bpm: float
    beat_morphology: Literal["wide_complex", "narrow_complex", "normal"]

    @field_validator("rr_interval_ms", "qrs_duration_ms")
    @classmethod
    def must_be_positive(cls, v):
        if v <= 0:
            raise ValueError("must be > 0")
        return v

    @field_validator("heart_rate_bpm")
    @classmethod
    def hr_in_range(cls, v):
        # Lower bound widened from an original 20 bpm after a real training-corpus
        # crash on genuine MIT-BIH data: a true ~3.7s inter-beat gap (16.14 bpm
        # instantaneous rate) was found in real Holter data, consistent with a
        # genuine sinus pause / conduction abnormality, not a computation bug --
        # AAMI_MAP already covers essentially every real beat symbol, so RR-interval
        # skips in data_prep.py should only ever skip genuine non-beat annotations.
        # 5 bpm (~12s gap) is a generous floor catching only truly degenerate values
        # (e.g. a stray non-positive interval), not real extreme-but-genuine pauses.
        if not (5 <= v <= 300):
            raise ValueError("heart_rate_bpm must be in [5, 300]")
        return v


class SegmentMetadata(BaseModel):
    window_samples: int
    sample_rate_hz: int
    lead: str
    signal_quality_index: float

    @field_validator("window_samples")
    @classmethod
    def window_samples_fixed(cls, v):
        if v != 360:
            raise ValueError("window_samples must be 360")
        return v

    @field_validator("sample_rate_hz")
    @classmethod
    def sample_rate_fixed(cls, v):
        if v != 360:
            raise ValueError("sample_rate_hz must be 360")
        return v

    @field_validator("signal_quality_index")
    @classmethod
    def sqi_in_range(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError("signal_quality_index must be in [0.0, 1.0]")
        return v


class ClinicalFlags(BaseModel):
    requires_urgent_review: bool
    flag_reason: str | None = None
    consecutive_abnormal_beats: int

    @field_validator("consecutive_abnormal_beats")
    @classmethod
    def non_negative(cls, v):
        if v < 0:
            raise ValueError("consecutive_abnormal_beats must be >= 0")
        return v

    @model_validator(mode="after")
    def flag_reason_consistency(self):
        if not self.requires_urgent_review and self.flag_reason is not None:
            raise ValueError("flag_reason must be null when requires_urgent_review is false")
        return self


class HealthEventJSON(BaseModel):
    event_id: str
    timestamp: str
    classification: ClassificationResult
    signal_features: SignalFeatures
    segment_metadata: SegmentMetadata
    clinical_flags: ClinicalFlags

    @field_validator("event_id")
    @classmethod
    def event_id_pattern(cls, v):
        # {seq:04d} means zero-padded to AT LEAST 4 digits — Python's format
        # spec grows past 4 digits for larger numbers rather than truncating,
        # so the pattern must allow 4+ digits, not exactly 4.
        if not re.match(r"^evt_\d+_\d{4,}$", v):
            raise ValueError(r"event_id must match pattern evt_{unix_ts}_{seq:04d}")
        return v
