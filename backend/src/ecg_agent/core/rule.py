"""The fixed triage rule over the sender's outputs (research repo: reasoning/training_targets.py).

In the paper this rule is the reference answer, so every receiver error belongs to the channel. In the agent it is
the safety guardrail: the receiver's tier may never be lower than the rule's.
"""
from __future__ import annotations

TIERS = ("routine", "priority", "urgent")
RANK = {t: i for i, t in enumerate(TIERS)}


def beat_tier(event: dict) -> str:
    if event["clinical_flags"]["requires_urgent_review"]:
        return "urgent"
    if event["classification"]["label"] == "N":
        return "routine"
    return "priority"


def window_tier(events: list[dict]) -> str:
    return max((beat_tier(e) for e in events), key=RANK.__getitem__, default="routine")


def most_urgent_index(events: list[dict]) -> int:
    ranks = [RANK[beat_tier(e)] for e in events]
    return ranks.index(max(ranks))


def guideline_fact(label: str) -> str:
    return {
        "N": "A normal sinus pattern calls for routine ongoing monitoring.",
        "S": "Frequent or sustained supraventricular ectopy warrants confirmation and clinical correlation.",
        "V": "Frequent ventricular ectopy or three or more beats in a row warrants prompt review.",
        "F": "Repeated fusion beats alongside abnormal beats require the same escalation as sustained ventricular ectopy.",
        "Q": "An uncertain classification requires repeat recording or manual review before interpretation.",
    }[label]


def rule_answer(events: list[dict]) -> dict:
    """The rule's own answer in the receiver's output schema (used by the offline receiver and as a fallback)."""
    i = most_urgent_index(events)
    top = events[i]
    tier, label, n = beat_tier(top), top["classification"]["label"], len(events)
    if tier == "routine":
        why = f"All {n} beats are normal, so routine ongoing monitoring is appropriate."
    elif tier == "urgent":
        run = top["clinical_flags"]["consecutive_abnormal_beats"]
        why = (f"Beat {i + 1} of {n} ends a run of {run} consecutive abnormal beats, which meets the urgent rule."
               if run >= 3 else f"Beat {i + 1} of {n} is a high-confidence {label}-class beat, which meets the urgent rule.")
    else:
        why = f"Beat {i + 1} of {n} is an isolated {label}-class beat that needs priority clinical correlation."
    return {"urgency_tier": tier, "justification": why, "referenced_guideline_fact": guideline_fact(label)}
