"""Schema-constrained greedy decoding for the receiver's JSON answer (vendored from the research repo).

A LogitsProcessor that makes every generation parse:

    {"urgency_tier":"<routine|priority|urgent>","justification":"<text>","referenced_guideline_fact":"<text>"}

The literal parts are forced as the tokenizer's own tokenisation of each literal (so
they always cost the same few tokens), the tier is restricted to the three valid words,
and the two free-text fields may contain any token without a double quote, backslash,
brace or control character, so they are always valid JSON strings that the project's
last-JSON-object extractor reads correctly. A field ends when the model emits any token
that starts the closing literal (e.g. '"' or '","'), or a merged token of safe text followed
by the start of it (e.g. '."', how the targets end: 'monitoring' + '."' + '}'); the rest of
the literal is then forced, or is closed after ``field_cap`` tokens; it cannot close before it holds a
non-whitespace token (the schema rejects empty fields). After the closing
brace only an end-of-sequence token is allowed. The model's own scores decide the tier,
the text and when to close a field. Used identically for every arm.

Each generate() call needs a fresh processor (it keeps per-row state across steps).
"""
import torch
from transformers import LogitsProcessor, StoppingCriteria

TIERS = ("routine", "priority", "urgent")
PARTS = ('{"urgency_tier":"', "<tier>", '","justification":"', "<text>",
         '","referenced_guideline_fact":"', "<text>", '"}', "<end>")
FORBIDDEN_IN_TEXT = '"\\{}'


class TokenTable:
    """Per-vocabulary lookups, built once per tokenizer (a few seconds for 262k tokens)."""

    def __init__(self, tokenizer, device="cpu"):
        n = len(tokenizer)
        self.strings = tokenizer.batch_decode([[i] for i in range(n)])
        special = set(tokenizer.all_special_ids)
        ok = [bool(s) and i not in special and not any(c in s for c in FORBIDDEN_IN_TEXT)
              and "�" not in s and all(ord(c) >= 32 for c in s) for i, s in enumerate(self.strings)]
        self.plain_ok = torch.tensor(ok, dtype=torch.bool, device=device)
        self.content_ok = self.plain_ok & torch.tensor([bool(s.strip()) for s in self.strings], device=device)
        self.size = n
        encode = lambda s: list(tokenizer(s, add_special_tokens=False)["input_ids"])
        self.canon = {p: encode(p) for p in PARTS if not p.startswith("<")}
        self.tiers = {w: encode(w) for w in TIERS}
        for p, ids in list(self.canon.items()) + list(self.tiers.items()):
            assert "".join(self.strings[i] for i in ids) == p, (p, ids)  # tokenisation round-trips
        # Ways to start closing a text field -> {first id: forced remainder}: a token spelling a
        # prefix of the closing literal, or safe text + such a prefix (merged, e.g. '."'), whose
        # remainder has a round-tripping tokenisation. closer_text[p][id] = the text part.
        by_string = {}
        for i, s in enumerate(self.strings):
            if i not in special and s:
                by_string.setdefault(s, []).append(i)
        safe = lambda t: not any(c in t for c in FORBIDDEN_IN_TEXT) and "�" not in t and all(ord(c) >= 32 for c in t)
        self.closers, self.closer_text = {}, {}
        for p in ('","referenced_guideline_fact":"', '"}'):
            opts, texts = {}, {}
            for i, s in enumerate(self.strings):
                if i in special or '"' not in s:
                    continue
                q = s.index('"')
                head, tail = s[:q], s[q:]
                if not safe(head) or not p.startswith(tail):
                    continue
                rest = encode(p[len(tail):]) if p[len(tail):] else []
                if "".join(self.strings[j] for j in rest) != p[len(tail):]:
                    continue
                opts[i], texts[i] = rest, head
            self.closers[p], self.closer_text[p] = opts, texts
        firsts = [ids[0] for ids in self.tiers.values()]
        assert len(set(firsts)) == len(firsts), "tier words must differ in their first token"


DECISION_PARTS = 2  # PARTS[0] (the opening literal) and PARTS[1] (the tier): all ranking needs


def decision_prefix(table: TokenTable, tier: str) -> list[int]:
    """Token ids of the answer up to and including the tier: '{"urgency_tier":"' + tier."""
    return table.canon[PARTS[0]] + table.tiers[tier]


class DecisionStop(StoppingCriteria):
    """Stopping criterion: end generation as soon as the tier word is complete (the answer's first ~7 tokens)."""

    def __init__(self, table: TokenTable, prompt_len: int):
        self.table, self.start, self.k = table, prompt_len, len(table.canon[PARTS[0]])
        self.first_to_len = {ids[0]: len(ids) for ids in table.tiers.values()}

    def __call__(self, input_ids, scores, **kwargs):
        gen = input_ids[0, self.start:]
        done = len(gen) > self.k and len(gen) >= self.k + self.first_to_len.get(int(gen[self.k]), 1)
        return torch.full((input_ids.shape[0],), done, dtype=torch.bool, device=input_ids.device)


class _Row:
    def __init__(self, table, start_part=0):
        self.table = table
        self.tier, self.hit_cap, self.done, self.has_content = None, False, False, False
        self._enter(start_part)

    def _enter(self, part):
        self.part, self.k, self.count, self.has_content = part, 0, 0, False
        self.seq = self.table.canon.get(PARTS[part])  # forced token sequence for a literal, else None

    def _step_forced(self):
        self.k += 1
        if self.k == len(self.seq):
            self._enter(self.part + 1)

    def consume(self, tid):
        p = PARTS[self.part]
        if p == "<end>":
            self.done = True
        elif p == "<tier>":
            if self.seq is None:  # the first tier token picks the tier
                self.tier = next(w for w, ids in self.table.tiers.items() if ids[0] == tid)
                self.seq = self.table.tiers[self.tier]
            self._step_forced()
        elif p == "<text>":
            rest = self.table.closers[PARTS[self.part + 1]].get(tid)
            if rest is not None:  # the model starts closing the field; force the remainder
                self._enter(self.part + 1)
                self.seq, self.k = rest, 0
                if not rest:
                    self._enter(self.part + 1)
            else:
                self.count += 1
                self.has_content |= bool(self.table.strings[tid].strip())
        else:
            self._step_forced()

    def allowed(self, device, eos_ids, field_cap):
        t = self.table
        a = torch.zeros(t.size, dtype=torch.bool, device=device)
        p = PARTS[self.part]
        if self.done or p == "<end>":
            a[eos_ids] = True
        elif p == "<tier>":
            a[[ids[0] for ids in t.tiers.values()] if self.seq is None else [self.seq[self.k]]] = True
        elif p == "<text>":
            closing = PARTS[self.part + 1]
            if not self.has_content:
                a |= (t.content_ok if self.count >= field_cap else t.plain_ok).to(device)
                # a field may not close empty; a merged closer whose text part has content may
                a[[i for i, h in t.closer_text[closing].items() if h.strip()]] = True
                return a
            if self.count >= field_cap:
                self.hit_cap = True  # only the closing literal is allowed now
            else:
                a |= t.plain_ok.to(device)
            a[list(t.closers[closing])] = True
        else:
            a[self.seq[self.k]] = True
        return a


class SchemaJsonProcessor(LogitsProcessor):
    """``start_part`` > 0 resumes the schema part-way: the earlier parts were placed in the prompt (explain-later)."""

    def __init__(self, table: TokenTable, eos_ids, field_cap=40, start_part=0):
        self.table, self.eos_ids, self.field_cap, self.start_part = table, list(eos_ids), field_cap, start_part
        self.rows, self.seen = None, 0

    def __call__(self, input_ids, scores):
        if self.rows is None:
            self.rows = [_Row(self.table, self.start_part) for _ in range(input_ids.shape[0])]
            self.seen = input_ids.shape[1]  # prompt length (0 when generating from inputs_embeds)
        for t in range(self.seen, input_ids.shape[1]):  # tokens generated since the last call
            for i, row in enumerate(self.rows):
                if not row.done:
                    row.consume(int(input_ids[i, t]))
        self.seen = input_ids.shape[1]
        mask = torch.stack([row.allowed(scores.device, self.eos_ids, self.field_cap) for row in self.rows])
        return scores.masked_fill(~mask, float("-inf"))

    def summary(self):
        """Per row: the tier chosen and whether a free-text field hit the cap."""
        return [{"tier": r.tier, "hit_field_cap": r.hit_cap} for r in (self.rows or [])]
