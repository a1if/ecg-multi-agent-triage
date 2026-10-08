"""Multi-event virtual-token adapter (research repo: reasoning/multi_event_adapter.py, virtual_adapter.py).

N sender vectors (32-d + 3 side inputs) -> N x k virtual tokens in Gemma's input-embedding space, with learned
per-slot position embeddings and one learnable norm scale. The frozen model sees text prefix + virtual tokens +
text suffix; Gemma 4's per-layer inputs for the virtual positions are built from PAD surrogates.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PLACEHOLDER = "\n[[ADAPTER_EVENT_DATA_REPLACED_BY_VIRTUAL_TOKENS]]\n"


def side_features(events: list[dict]) -> np.ndarray:
    """(n, 3) heart rate, RR interval and run length, scaled to O(1) (recipe r4 side inputs)."""
    out = np.empty((len(events), 3), dtype=np.float32)
    for i, e in enumerate(events):
        sf, cf = e["signal_features"], e["clinical_flags"]
        out[i, 0] = np.clip((sf["heart_rate_bpm"] - 80.0) / 30.0, -3, 3)
        out[i, 1] = np.clip((sf["rr_interval_ms"] - 750.0) / 250.0, -3, 3)
        out[i, 2] = np.log1p(min(cf["consecutive_abnormal_beats"], 20)) / np.log(4.0)
    return out


def adapter_inputs(vectors: np.ndarray, events: list[dict], input_dim: int) -> np.ndarray:
    v = np.asarray(vectors, dtype=np.float32)
    return v if input_dim == v.shape[1] else np.concatenate([v, side_features(events)], axis=1)


class MultiEventVirtualAdapter(nn.Module):
    def __init__(self, embedding_dim: int, *, num_tokens: int = 4, input_dim: int = 35, max_events: int = 50,
                 init_scale: float = 1.0):
        super().__init__()
        self.embedding_dim, self.num_tokens, self.input_dim, self.max_events = (embedding_dim, num_tokens,
                                                                                input_dim, max_events)
        self.projection = nn.Linear(input_dim, num_tokens * embedding_dim)
        self.position = nn.Parameter(torch.zeros(max_events, num_tokens, embedding_dim))
        self.log_scale = nn.Parameter(torch.tensor(math.log(init_scale)))

    @classmethod
    def from_state_dict(cls, sd: dict) -> MultiEventVirtualAdapter:
        max_events, k, dim = sd["position"].shape
        m = cls(dim, num_tokens=k, input_dim=sd["projection.weight"].shape[1], max_events=max_events)
        m.load_state_dict(sd)
        return m.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(N, D) or (B, N, D) -> (B, N * k, E)."""
        if x.ndim == 2:
            x = x.unsqueeze(0)
        b, n, d = x.shape
        if d != self.input_dim or n > self.max_events:
            raise ValueError(f"expected (B, N <= {self.max_events}, {self.input_dim}), got {tuple(x.shape)}")
        y = self.projection(x).view(b, n, self.num_tokens, self.embedding_dim) + self.position[:n].unsqueeze(0)
        y = F.normalize(y, dim=-1) * self.log_scale.exp()
        return y.reshape(b, n * self.num_tokens, self.embedding_dim)


@dataclass(frozen=True)
class AdapterInputs:
    inputs_embeds: torch.Tensor
    attention_mask: torch.Tensor
    per_layer_inputs: torch.Tensor


def split_prompt_ids(processor, prefix: str, suffix: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Render the chat prompt once with a placeholder and cut its tokens out (keeps boundary merges identical)."""
    rendered = processor.apply_chat_template([{"role": "user", "content": prefix + PLACEHOLDER + suffix}],
                                             add_generation_prompt=True, tokenize=False)
    start = rendered.find(PLACEHOLDER)
    if start < 0:
        raise RuntimeError("adapter placeholder lost in the chat template")
    end = start + len(PLACEHOLDER)
    enc = processor.tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True, return_tensors="pt")
    overlap = [i for i, (a, b) in enumerate(enc["offset_mapping"][0].tolist()) if a < end and b > start]
    if not overlap or overlap != list(range(overlap[0], overlap[-1] + 1)):
        raise RuntimeError("adapter placeholder tokens missing or non-contiguous")
    ids = enc["input_ids"]
    return ids[:, :overlap[0]], ids[:, overlap[-1] + 1:]


def compose_inputs(model, adapter: MultiEventVirtualAdapter, x: np.ndarray, prefix_ids: torch.Tensor,
                   suffix_ids: torch.Tensor) -> AdapterInputs:
    emb = model.get_input_embeddings()
    device = emb.weight.device
    pre, suf = emb(prefix_ids.to(device)), emb(suffix_ids.to(device))
    virtual = adapter(torch.as_tensor(x, dtype=torch.float32, device=next(adapter.parameters()).device))
    virtual = virtual.to(device=device, dtype=pre.dtype)
    inputs_embeds = torch.cat((pre, virtual, suf), dim=1)
    mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)
    pad = model.config.text_config.pad_token_id
    surrogate = torch.cat((prefix_ids.to(device),
                           torch.full((1, virtual.shape[1]), pad, dtype=prefix_ids.dtype, device=device),
                           suffix_ids.to(device)), dim=1)
    lm = getattr(model, "language_model", None) or model.model.language_model
    return AdapterInputs(inputs_embeds, mask, lm.get_per_layer_inputs(surrogate, None))
