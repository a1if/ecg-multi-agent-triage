"""Clinician questions (design §6c): understand the question, find what it refers to, decide if it is in scope.

Three layers, cheapest and safest first:

1. ``safety_screen`` — fixed patterns, checked in code before any LLM sees the question. Requests for diagnosis,
   treatment, medication or prognosis, requests to change a tier, and attempts to override the instructions get a
   fixed reply. Defence in depth: even a question that would fool the classifier never reaches it.
2. ``find_refs`` — window ids ("w005") and times ("at 1:32", "around 90 s", "minute 3") are parsed in code and mapped
   to windows, so the LLM never has to get an id or a number right.
3. A classifier maps the question onto a fixed list of types: ``GemmaClassifier`` (JSON, validated in code) with
   ``RuleClassifier`` (keywords) as its fallback and as the offline classifier.

The answer itself is composed in code from evidence (see ReasoningAgent); the LLM never writes free text to the
clinician except explanations, which pass the grounding check.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ecg_agent.core.prompts import extract_last_json_object
from ecg_agent.receiver.base import GenerateRequest

TYPES = {
    "explain": "why a reviewed window got its tier",
    "evidence": "show the beats of a window (classes, confidences, runs)",
    "review": "review a part of the recording that was not reviewed yet",
    "compare": "how the channels (compact, filtered, adapter) did, where answers disagreed",
    "system": "how this system works",
    "clinical": "asks for medical advice: diagnosis, treatment, medication, risk, danger, or what it means for the patient",
    "change_request": "asks to change, mark, approve or dismiss a result",
    "unclear": "none of the above, or the question does not say which part of the recording",
}
# Layer 2: questions the patterns missed but the classifier recognises as out of scope -> REFUSALS key
REFUSED_TYPES = {"clinical": "clinical", "change_request": "change_tier"}

REFUSALS = {
    "clinical": ("I can't advise on diagnosis, treatment, medication or prognosis: that is outside what this research "
                 "prototype is for. I can explain a finding, show the beats behind it, or review another part of the "
                 "recording."),
    "change_tier": ("Tiers are not changed through questions. If you disagree with a finding, use the override "
                    "control on that window; the change is recorded with your name and reason."),
    "other_patient": "I can only answer about this recording.",
    "budget": "The question limit for this run has been reached.",
}

# Layer 1. The second half of each pattern was added after a real run let these through to the classifier:
# "Could these ventricular beats be dangerous for her?" and "Mark everything as fine please".
_CLINICAL = re.compile(
    r"\b(treat\w*|medicat\w*|drugs?|dos(e|age|ing)|prescri\w*|diagnos\w*|prognos\w*|surgery|ablation|pacemaker|"
    r"defibrillat\w*|amiodarone|beta.?blockers?|anticoagula\w*|life expectancy|"
    r"should (i|we|the patient) (take|start|stop|be)|"
    r"dangerous|danger|risk\w*|harm\w*|serious|worr(y|ied)|fatal|die|dying|safe for (him|her|them|the patient))\b",
    re.I)
_CHANGE = re.compile(
    r"\b(change|set|mark|downgrade|lower|upgrade|raise|override|make|reclassify)\b[^.?!]*\b(tier|routine|priority|urgent)\b"
    r"|\b(mark|set|call|label|sign off|approve|dismiss|clear)\b[^.?!]*\b(fine|normal|ok(ay)?|safe|benign|nothing)\b"
    r"|\bignore\b[^.?!]*\b(instructions?|rules?|guardrails?|previous|above)\b", re.I)
_OTHER = re.compile(r"\b(other|another|different) (patient|record(ing)?)\b", re.I)


def safety_screen(question: str) -> str | None:
    """A refusal key if the question is out of scope, else None. Runs before any LLM."""
    if _CHANGE.search(question):
        return "change_tier"
    if _CLINICAL.search(question):
        return "clinical"
    if _OTHER.search(question):
        return "other_patient"
    return None


_WID = re.compile(r"\bw(\d{1,3})\b", re.I)
_CLOCK = re.compile(r"\b(\d{1,2}):(\d{2})\b")
_SECONDS = re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:s|sec|secs|seconds?)\b", re.I)
_MINUTE = re.compile(r"\bminute\s+(\d+)\b", re.I)


def find_refs(question: str, candidates: list[dict]) -> list[str]:
    """Window ids the question refers to, in order: explicit ids, then times mapped to the window containing them."""
    known = {w["id"] for w in candidates}
    out = [f"w{int(m):03d}" for m in _WID.findall(question) if f"w{int(m):03d}" in known]
    times = [60 * int(a) + int(b) for a, b in _CLOCK.findall(question)]
    times += [float(s) for s in _SECONDS.findall(question)] + [60.0 * int(m) for m in _MINUTE.findall(question)]
    for t in times:
        hit = min(candidates, key=lambda w: 0 if w["t_start_s"] <= t <= w["t_end_s"]
                  else min(abs(t - w["t_start_s"]), abs(t - w["t_end_s"])), default=None)
        if hit and hit["id"] not in out:
            out.append(hit["id"])
    return out


@dataclass
class Intent:
    type: str
    window_id: str | None = None
    refusal: str | None = None  # a REFUSALS key: the answer is fixed and nothing else happens
    source: str = "rule"
    notes: list[str] = field(default_factory=list)


class RuleClassifier:
    name = "rule"

    async def classify(self, question: str, view: dict) -> Intent:
        q = question.lower()
        refs = find_refs(question, view["candidates"])
        wid = refs[0] if refs else None
        reviewed = wid in view["reviewed"] if wid else False
        if re.search(r"\b(channel|adapter|filtered|compact|disagree\w*)\b", q):
            return Intent("compare", wid)
        if re.search(r"\b(how (does|do) (this|it|you) work|what is this system|how was this (made|decided))\b", q):
            return Intent("system")
        if re.search(r"\b(show|list|which beats|evidence|how many)\b", q) and wid:
            return Intent("evidence", wid)
        if re.search(r"\b(look at|review|check|examine|what about|anything (at|around|in))\b", q) and wid:
            return Intent("explain" if reviewed else "review", wid)
        if re.search(r"\b(why|explain|reason|justif\w*)\b", q) and wid:
            return Intent("explain" if reviewed else "review", wid)
        if wid:
            return Intent("explain" if reviewed else "review", wid)
        return Intent("unclear")


CLASSIFY_SYSTEM = ("You route a clinician's question about one automated ECG triage run. Reply with ONE JSON object "
                   'and nothing else: {"type": "<type>", "window_id": "<id or null>"}.')


class GemmaClassifier:
    """The LLM picks the type; code validates it and fills in the window from ``find_refs``. Any problem falls back
    to the rule classifier, so a question always gets routed."""

    name = "gemma"

    def __init__(self, receiver, fallback: RuleClassifier | None = None):
        self.receiver, self.fallback = receiver, fallback or RuleClassifier()

    async def classify(self, question: str, view: dict) -> Intent:
        refs = find_refs(question, view["candidates"])
        types = "\n".join(f"  {k}: {v}" for k, v in TYPES.items())
        windows = ", ".join(f"{w['id']} ({w['t_start_s']:.0f}-{w['t_end_s']:.0f} s, "
                            f"{'reviewed' if w['id'] in view['reviewed'] else 'not reviewed'})"
                            for w in view["candidates"][:30])
        prompt = (f"Types:\n{types}\n\nWindows: {windows}\nWindows the question names: {refs or 'none'}\n\n"
                  f"Question: {question[:500]}\n\nYour JSON:")
        try:
            res = await self.receiver.generate(GenerateRequest(prompt=prompt, system=CLASSIFY_SYSTEM, max_new_tokens=40))
            obj = extract_last_json_object(res.text) if res else None
            if not obj or obj.get("type") not in TYPES:
                raise ValueError(f"unusable classification {obj!r}")
        except Exception as exc:  # noqa: BLE001 - routing must always succeed
            intent = await self.fallback.classify(question, view)
            intent.notes.append(f"LLM classifier unavailable or unusable ({type(exc).__name__}); keyword routing")
            return intent
        wid = obj.get("window_id")
        if wid not in {w["id"] for w in view["candidates"]}:
            wid = refs[0] if refs else None  # never trust an id the code did not find or verify
        t = obj["type"]
        if t in ("explain", "review") and wid:  # the agent's own records decide which of the two it is
            t = "explain" if wid in view["reviewed"] else "review"
        if t in ("explain", "review", "evidence") and not wid:
            t = "unclear"
        return Intent(t, wid, source="gemma")
