"""Receiver prompts for the three channels, byte-identical to the paper's item-7 prompts.

Every channel shares the system prompt, the class-neutral background context and the output instructions; only the
event data differs: every beat as compact JSON, only the non-normal beats (filtered), or N x 4 virtual tokens.
"""
from __future__ import annotations

import json

CHANNELS = ("compact", "filtered", "adapter")

SYSTEM_PROMPT = (
    "You are a clinical decision support assistant reviewing a single ECG beat "
    "classification. You will be given structured signal features and fixed "
    "background context for the detected beat class. Based on this information "
    "only, output a structured urgency tier and a brief justification. Do not "
    "invent information not present in the input. This is a research prototype; "
    "your output is not sanctioned clinical advice."
)

OUTPUT_INSTRUCTIONS = (
    "Respond with a JSON object with exactly three fields:\n"
    '  "urgency_tier": one of "routine", "priority", "urgent". Apply this rule '
    "exactly: use \"routine\" ONLY for a normal (N-class) beat. Use \"urgent\" if "
    "EITHER of these is true on its own, independently — each is sufficient by "
    "itself, do not require both: (a) consecutive_abnormal_beats is 3 or more, "
    "OR (b) this is a ventricular or fusion beat with confidence greater than "
    "0.85. Criterion (b) alone is sufficient for \"urgent\" even when "
    "consecutive_abnormal_beats is only 1 — do not downgrade to \"priority\" "
    "in that case just because the background context describes an isolated "
    "occurrence as usually benign; confidence above 0.85 overrides that framing. "
    "Use \"priority\" only for a non-normal beat that meets NEITHER urgent "
    "criterion above — a non-normal beat is never \"routine\".\n"
    '  "justification": a one-sentence clinical rationale for the tier\n'
    '  "referenced_guideline_fact": the specific fact from the background '
    "context you relied on, stated in your own words\n"
    "Your justification must explicitly draw on the background context provided."
)

NEUTRAL_CONTEXT = (
    "This event was classified using AAMI beat-type criteria (normal, "
    "supraventricular ectopic, ventricular ectopic, fusion, or unclassifiable). "
    "General escalation criteria, across all classes: a normal sinus beat needs "
    "only routine ongoing monitoring. An isolated supraventricular, ventricular, "
    "or fusion beat is usually benign on its own, but frequent occurrences, "
    "three or more consecutive abnormal beats, or a high-confidence ventricular "
    "or fusion classification each independently warrant prompt clinical review. "
    "A beat that cannot be classified due to low signal quality cannot be "
    "interpreted automatically and requires manual review rather than an "
    "automated judgement. Use the event representation below to determine "
    "which of these situations applies to this specific beat."
)


def compact_payload(event: dict) -> dict:
    c = event["classification"]
    return {"label": c["label"], "confidence": c["confidence"],
            "consecutive_abnormal_beats": event["clinical_flags"]["consecutive_abnormal_beats"],
            "signal_quality_index": event["segment_metadata"]["signal_quality_index"]}


def _note(n: int) -> str:
    return (f"The event data below contains {n} consecutive beats from one recording. Apply the rule to each "
            "beat and report the single most urgent tier among them.\n")


def _filtered_note(n: int) -> str:
    return (f"The event data below summarises {n} consecutive beats from one recording. Normal (N) beats are not "
            "listed individually: their number is given as normal_beats_not_listed. Every non-normal beat is listed "
            "with its position in the sequence (1 = first beat). Apply the rule to each listed beat and report the "
            f"single most urgent tier among all {n} beats; if no beat is listed, every beat is normal.\n")


def scaffold_parts(n: int) -> tuple[str, str]:
    """(prefix, suffix) around the event payload; the adapter's virtual tokens go between them."""
    prefix = f"{SYSTEM_PROMPT}\n\n--- Background context ---\n{NEUTRAL_CONTEXT}\n{_note(n)}--- Event data ---\n"
    return prefix, f"\n\n--- Instructions ---\n{OUTPUT_INSTRUCTIONS}"


def compact_prompt(events: list[dict]) -> str:
    pre, suf = scaffold_parts(len(events))
    return pre + json.dumps([compact_payload(e) for e in events], indent=2) + suf


def filtered_prompt(events: list[dict]) -> str:
    n = len(events)
    listed = [{"position": k + 1, **compact_payload(e)} for k, e in enumerate(events)
              if e["classification"]["label"] != "N"]
    payload = json.dumps({"total_beats": n, "normal_beats_not_listed": n - len(listed), "non_normal_beats": listed},
                         indent=2)
    return (f"{SYSTEM_PROMPT}\n\n--- Background context ---\n{NEUTRAL_CONTEXT}\n{_filtered_note(n)}--- Event data ---\n"
            f"{payload}\n\n--- Instructions ---\n{OUTPUT_INSTRUCTIONS}")


def extract_last_json_object(text: str) -> dict:
    """The last balanced {...} block in ``text``, parsed."""
    depth, end, start = 0, None, None
    for i in range(len(text) - 1, -1, -1):
        if text[i] == "}":
            if depth == 0:
                end = i
            depth += 1
        elif text[i] == "{":
            depth -= 1
            if depth == 0 and end is not None:
                start = i
                break
    if start is None or end is None:
        raise ValueError(f"no balanced JSON object in output: {text[:200]!r}")
    return json.loads(text[start:end + 1])
