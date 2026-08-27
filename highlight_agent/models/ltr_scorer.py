from __future__ import annotations

import typing
from pathlib import Path

import torch
import torch.nn as nn

class AdditiveAttentionScorer(nn.Module):
    def __init__(self, in_features: int = 7, hidden_dim: int = 32):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        
        self.proj = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.Tanh()
        )
        self.attn_head = nn.Linear(hidden_dim, 1, bias=False)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return self.attn_head(x)

    def save(self, path: str | Path, metadata: dict | None = None) -> None:
        torch.save({
            "state_dict": self.state_dict(),
            "in_features": self.in_features,
            "hidden_dim": self.hidden_dim,
            "metadata": metadata or {}
        }, path)

    @classmethod
    def load(cls, path: str | Path) -> AdditiveAttentionScorer:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(in_features=ckpt["in_features"], hidden_dim=ckpt["hidden_dim"])
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model
