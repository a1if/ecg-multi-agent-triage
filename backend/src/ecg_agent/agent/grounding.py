"""Grounding check for LLM-written explanations (design §5.3).

An explanation may only state numbers that exist in the evidence it was written from: beat positions in the window,
the window size, run lengths, class counts, the sender's confidences, heart rate and RR values, and the rule's own
thresholds. Anything else ("3-15/25:19999999999999", "2032") is a sign the text is not about this window, so it is
not shown; the rule's own wording is shown instead, labelled.

This is deliberately strict and simple: a false alarm costs a less fluent sentence, a miss costs a clinician
reading a made-up number.
"""
from __future__ import annotations

import json
import re
from collections import Counter

_NUM = re.compile(r"\d+(?:\.\d+)?")
RULE_NUMBERS = {"0.85", "85", "3", "1"}  # thresholds and "1 beat" phrasing the instructions themselves use


def _forms(x: float) -> set[str]:
    """The ways a number from the evidence may legitimately be written."""
    out = {f"{x:.0f}", f"{x:.1f}", f"{x:.2f}", f"{x:.3f}"}
    if 0 <= x <= 1:
        out |= {f"{100 * x:.0f}", f"{100 * x:.1f}"}  # a confidence as a percentage
    return {s.rstrip("0").rstrip(".") if "." in s else s for s in out} | out


def allowed_numbers(events: list[dict]) -> set[str]:
    n = len(events)
    ok = set(RULE_NUMBERS) | {str(i) for i in range(1, n + 1)}
    for count in Counter(e["classification"]["label"] for e in events).values():
        ok.add(str(count))
    for e in events:
        ok |= _forms(e["classification"]["confidence"])
        ok |= _forms(e["signal_features"]["heart_rate_bpm"]) | _forms(e["signal_features"]["rr_interval_ms"])
        ok.add(str(e["clinical_flags"]["consecutive_abnormal_beats"]))
    return ok


_WINDOW_ID = re.compile(r"\bw\d{3}\b", re.I)
_TIER = re.compile(r"\b(routine|priority|urgent)\b", re.I)
_NO_REVIEW = re.compile(r"\b(no|none of the|zero) (windows?|segments?|parts?)\b[^.]*\breview", re.I)


def _fact_numbers(obj, out: set[str]) -> set[str]:
    """Every number anywhere in a facts structure, in the forms it may be written."""
    if isinstance(obj, bool):
        return out
    if isinstance(obj, int | float):
        out |= _forms(float(obj))
        if float(obj).is_integer():
            out.add(str(int(obj)))
    elif isinstance(obj, dict):
        for v in obj.values():
            _fact_numbers(v, out)
    elif isinstance(obj, list | tuple):
        for v in obj:
            _fact_numbers(v, out)
        out.add(str(len(obj)))  # "3 windows were reviewed" counts a list
    return out


def check_narrative(text: str, facts: dict) -> list[str]:
    """Problems with an LLM-written run summary (empty list = grounded):

    * a number that is not in the facts (window ids like w003 are names, not numbers);
    * a tier that no fact carries (e.g. "routine" when nothing was routine);
    * saying no window needs review when some do;
    * a missing disclaimer."""
    problems = []
    ok = set(RULE_NUMBERS) | _fact_numbers(facts, set())
    bad = [m for m in _NUM.findall(_WINDOW_ID.sub(" ", text))
           if m not in ok and not ("." in m and m.rstrip("0").rstrip(".") in ok)]
    if bad:
        problems.append(f"numbers not in the facts: {bad[:5]}")
    tiers = {t.lower() for t in _TIER.findall(json.dumps(facts))}
    extra = {t.lower() for t in _TIER.findall(text)} - tiers
    if extra:
        problems.append(f"tiers not in the facts: {sorted(extra)}")
    if facts.get("windows_needing_human_review") and _NO_REVIEW.search(text):
        problems.append("says no window needs review, but some do")
    if "not clinical advice" not in text.lower():
        problems.append("missing disclaimer")
    return problems


def check(text: str, events: list[dict]) -> list[str]:
    """Numbers in ``text`` that the evidence does not contain (empty list = grounded)."""
    ok = allowed_numbers(events)
    # "0.850" may match "0.85", but trailing zeros of a whole number are significant ("100" is not "1")
    return [m for m in _NUM.findall(text)
            if m not in ok and not ("." in m and m.rstrip("0").rstrip(".") in ok)]
