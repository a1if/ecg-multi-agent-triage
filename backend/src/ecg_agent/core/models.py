"""Sender networks. Vendored from the research repo (perception/model.py, model_rr.py) unchanged in behaviour.

CNN-LSTM over one 1-s, 360-sample, z-scored beat window, with a small RR-interval branch appended to every
timestep entering the context LSTM. The context LSTM's 32-d final hidden state is both the classifier input and
the vector the adapter turns into virtual tokens.
"""
import json
from pathlib import Path

import torch
import torch.nn as nn

from ecg_agent.core.rr_features import N_RR_FEATURES

AAMI_CLASSES = ["N", "S", "V", "F", "Q"]
RR_EMBED_DIM = 16


def _block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv1d(c_in, c_out, kernel_size=5, padding=2), nn.BatchNorm1d(c_out), nn.ReLU(),
                         nn.MaxPool1d(kernel_size=2, stride=2))


class CNNLSTMRR(nn.Module):
    """Input: x (B, 1, 360), rr (B, 5) standardised RR features. Output: (logits (B, 5), context (B, 32))."""

    def __init__(self):
        super().__init__()
        self.block1, self.block2, self.block3 = _block(1, 32), _block(32, 64), _block(64, 128)
        self.bilstm = nn.LSTM(input_size=128, hidden_size=64, batch_first=True, bidirectional=True)
        self.rr_embed = nn.Sequential(nn.Linear(N_RR_FEATURES, RR_EMBED_DIM), nn.ReLU())
        self.context_lstm = nn.LSTM(input_size=128 + RR_EMBED_DIM, hidden_size=32, batch_first=True)
        self.head = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 5))

    def forward(self, x: torch.Tensor, rr: torch.Tensor):
        x = self.block3(self.block2(self.block1(x)))  # (B, 128, 45)
        seq, _ = self.bilstm(x.permute(0, 2, 1))  # (B, 45, 128)
        rr_emb = self.rr_embed(rr).unsqueeze(1).expand(-1, seq.size(1), -1)
        _, (h_n, _) = self.context_lstm(torch.cat([seq, rr_emb], dim=2))
        context = h_n.squeeze(0)
        return self.head(context), context


def deployed_adapter(models_dir: str | Path) -> tuple[Path, str]:
    """The adapter deployment uses: the file and SHA-256 that ``manifest.json`` pins. Everything that loads or keys on
    the adapter goes through here, so a promotion is a manifest change and nothing else."""
    m = json.loads((Path(models_dir) / "manifest.json").read_text())["adapter"]
    return Path(models_dir) / m["file"], m["sha256"]
