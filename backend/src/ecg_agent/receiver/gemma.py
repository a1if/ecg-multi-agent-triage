"""In-process Gemma 4 E4B receiver (GPU). Used by the inference service only.

The receiver path is the paper's: 4-bit NF4 weights, greedy decoding under the schema-constrained logits processor,
and for the adapter channel the prompt embeddings are text prefix + N x 4 virtual tokens + text suffix.

Decide early, explain later (design §6d): ``triage(mode="decision")`` stops as soon as the tier word exists (about 7
answer tokens instead of 50-100); ``explain`` writes the justification later, on demand, with the answer's first
tokens placed in the prompt so they are read in parallel instead of generated again one by one.
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import numpy as np
import torch
from transformers import LogitsProcessor, StoppingCriteriaList

from ecg_agent.core.adapter import MultiEventVirtualAdapter, adapter_inputs, compose_inputs, split_prompt_ids
from ecg_agent.core.decoding import (
    DECISION_PARTS,
    PARTS,
    DecisionStop,
    SchemaJsonProcessor,
    TokenTable,
    decision_prefix,
)
from ecg_agent.core.prompts import compact_prompt, extract_last_json_object, filtered_prompt, scaffold_parts
from ecg_agent.core.rule import TIERS
from ecg_agent.receiver.base import GenerateRequest, GenerateResult, TriageRequest, TriageResult

MODEL_ID = os.environ.get("GEMMA_MODEL_ID", "google/gemma-4-E4B-it")
MAX_NEW, FIELD_CAP = 128, 40
OPEN = '{"urgency_tier":"'


class _StepClock(LogitsProcessor):
    """Records when each decoding step starts. Step 0 starts when the prompt has been processed (time to first
    token); step k + 1 starts once token k has been produced, so the tier token's time is known too."""

    def __init__(self):
        self.stamps: list[float] = []

    def __call__(self, input_ids, scores):
        torch.cuda.synchronize()
        self.stamps.append(time.perf_counter())
        return scores


class GemmaEngine:
    def __init__(self, adapter_checkpoint: str | Path, load_in_4bit: bool = True):
        from transformers import AutoProcessor, BitsAndBytesConfig, Gemma4ForConditionalGeneration

        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_compute_dtype=torch.bfloat16) if load_in_4bit else None
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.model = Gemma4ForConditionalGeneration.from_pretrained(
            MODEL_ID, quantization_config=quant, device_map="cuda", attn_implementation="sdpa")
        self.model.eval()
        self.tok = self.processor.tokenizer
        self.device = self.model.get_input_embeddings().weight.device
        self.table = TokenTable(self.tok, device=self.device)
        eos = self.model.generation_config.eos_token_id
        self.eos = list(eos) if isinstance(eos, (list, tuple)) else [eos]
        ck = torch.load(adapter_checkpoint, map_location="cpu", weights_only=False)
        self.adapter = MultiEventVirtualAdapter.from_state_dict(ck["adapter_state_dict"]).to(self.device)
        self.lock = threading.Lock()  # one GPU, one generation at a time
        self._scaffold: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._warm()

    @property
    def max_events(self) -> int:
        return self.adapter.max_events

    def _warm(self) -> None:
        ids = self._chat_ids("Hello")
        with torch.no_grad():
            self.model.generate(**ids, do_sample=False, max_new_tokens=4)

    def _chat_ids(self, content: str, system: str | None = None) -> dict:
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": content}]
        enc = self.processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
                                                 return_tensors="pt")
        return {"input_ids": enc["input_ids"].to(self.device), "attention_mask": enc["attention_mask"].to(self.device)}

    def _run(self, inputs: dict, constrained: bool, max_new: int, start_part: int = 0, stop_at_tier: bool = False):
        """-> (text, generated tokens, total ms, time to first token ms, time to decision ms or None).
        ``stop_at_tier`` ends generation once the tier word exists; ``start_part`` resumes the schema after parts
        that are already in the prompt."""
        clock = _StepClock()
        procs = [clock] + ([SchemaJsonProcessor(self.table, self.eos, FIELD_CAP, start_part)] if constrained else [])
        start = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
        stops = StoppingCriteriaList([DecisionStop(self.table, start)]) if stop_at_tier else None
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.model.generate(**inputs, do_sample=False, max_new_tokens=max_new, logits_processor=procs,
                                      stopping_criteria=stops)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        gen = out[0][start:]
        n_gen = int((gen != self.tok.pad_token_id).sum())
        st = clock.stamps
        ttft = ((st[0] if st else t1) - t0) * 1e3
        decision = None
        if constrained and start_part == 0:
            k = len(self.table.canon[PARTS[0]]) + 1  # step at which the tier token has been produced
            decision = ((st[k] if k < len(st) else t1) - t0) * 1e3
        return self.tok.decode(gen, skip_special_tokens=True), n_gen, (t1 - t0) * 1e3, ttft, decision

    def _inputs(self, req: TriageRequest, forced: list[int] | None = None) -> tuple[dict, int]:
        """Generation inputs for one window, with ``forced`` answer tokens appended after the generation prompt.
        Returns (inputs, prompt tokens excluding the forced ones)."""
        n = len(req.events)
        forced = forced or []
        if req.channel == "adapter":
            if req.vectors is None or len(req.vectors) != n:
                raise ValueError("the adapter channel needs one sender vector per event")
            if n > self.max_events:
                raise ValueError(f"the adapter carries at most {self.max_events} events")
            if n not in self._scaffold:
                self._scaffold[n] = split_prompt_ids(self.processor, *scaffold_parts(n))
            x = adapter_inputs(np.asarray(req.vectors, dtype=np.float32), req.events, self.adapter.input_dim)
            pre, suf = self._scaffold[n]
            suf = torch.cat([suf, torch.tensor([forced], dtype=suf.dtype)], dim=1)
            with torch.no_grad():
                ai = compose_inputs(self.model, self.adapter, x, pre, suf)
            inputs = {"inputs_embeds": ai.inputs_embeds, "attention_mask": ai.attention_mask,
                      "per_layer_inputs": ai.per_layer_inputs}
            return inputs, int(ai.attention_mask.shape[1]) - len(forced)
        text = (compact_prompt if req.channel == "compact" else filtered_prompt)(req.events)
        inputs = self._chat_ids(text)
        n_prompt = int(inputs["input_ids"].shape[1])
        if forced:
            f = torch.tensor([forced], dtype=inputs["input_ids"].dtype, device=self.device)
            inputs = {"input_ids": torch.cat([inputs["input_ids"], f], dim=1),
                      "attention_mask": torch.cat([inputs["attention_mask"], torch.ones_like(f)], dim=1)}
        return inputs, n_prompt

    def triage(self, req: TriageRequest) -> TriageResult:
        """``req.mode == "decision"``: stop at the tier. ``"full"``: also write the justification (the paper's path)."""
        decide = req.mode == "decision"
        with self.lock:
            inputs, n_prompt = self._inputs(req)
            raw, n_gen, ms, ttft, decision = self._run(inputs, constrained=True, max_new=MAX_NEW,
                                                       stop_at_tier=decide)
        base = dict(channel=req.channel, prompt_tokens=n_prompt, generated_tokens=n_gen, latency_ms=ms, ttft_ms=ttft,
                    decision_ms=decision, raw=raw, mode=req.mode)
        if decide:
            tier = next((t for t in TIERS if raw.startswith(OPEN + t)), None)
            return TriageResult(tier=tier, parsed=tier is not None, **base)
        try:
            ans = extract_last_json_object(raw)
            return TriageResult(tier=ans["urgency_tier"], justification=ans["justification"],
                                guideline_fact=ans["referenced_guideline_fact"], **base)
        except (ValueError, KeyError):
            return TriageResult(tier=None, parsed=False, **base)

    def explain(self, req: TriageRequest, tier: str) -> TriageResult:
        """Justification for a tier already decided. Greedy decoding continues as the full answer would have, up to
        floating-point differences between reading tokens in parallel and generating them one at a time."""
        if tier not in TIERS:
            raise ValueError(f"unknown tier {tier!r}")
        with self.lock:
            inputs, n_prompt = self._inputs(req, decision_prefix(self.table, tier))
            raw, n_gen, ms, ttft, _ = self._run(inputs, constrained=True, max_new=MAX_NEW, start_part=DECISION_PARTS)
        full = OPEN + tier + raw
        base = dict(channel=req.channel, prompt_tokens=n_prompt, generated_tokens=n_gen, latency_ms=ms, ttft_ms=ttft,
                    raw=full, mode="explain")
        try:
            ans = extract_last_json_object(full)
            return TriageResult(tier=tier, justification=ans["justification"],
                                guideline_fact=ans["referenced_guideline_fact"], **base)
        except (ValueError, KeyError):
            return TriageResult(tier=tier, parsed=False, **base)

    def generate(self, req: GenerateRequest) -> GenerateResult:
        with self.lock:
            inputs = self._chat_ids(req.prompt, req.system)
            text, n_gen, ms, _, _ = self._run(inputs, constrained=False, max_new=req.max_new_tokens)
        return GenerateResult(text=text, prompt_tokens=int(inputs["input_ids"].shape[1]), generated_tokens=n_gen,
                              latency_ms=ms)
