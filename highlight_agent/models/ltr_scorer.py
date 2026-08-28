from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class AdditiveAttentionScorer(nn.Module):
    """Small additive-attention scorer for seven-channel window features."""

    def __init__(self, in_features: int = 7, hidden_dim: int = 32):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim

        self.proj = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.Tanh(),
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

    def save(self, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
        """Save model weights and reproducibility metadata."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "checkpoint_version": "1.0",
                "state_dict": self.state_dict(),
                "in_features": self.in_features,
                "hidden_dim": self.hidden_dim,
                "metadata": metadata or {},
            },
            output_path,
        )

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        *,
        device: str | torch.device | None = None,
        expected_in_features: int | None = 7,
    ) -> tuple[AdditiveAttentionScorer, dict[str, Any]]:
        """Load a validated checkpoint and place the model on ``device``."""

        target_device = torch.device(device or "cpu")
        checkpoint = torch.load(path, map_location=target_device, weights_only=False)
        if not isinstance(checkpoint, dict):
            raise ValueError("LTR checkpoint must be a dictionary")

        required = {"state_dict", "in_features", "hidden_dim"}
        missing = sorted(required.difference(checkpoint))
        if missing:
            raise ValueError(f"LTR checkpoint is missing fields: {', '.join(missing)}")

        in_features = int(checkpoint["in_features"])
        hidden_dim = int(checkpoint["hidden_dim"])
        if expected_in_features is not None and in_features != expected_in_features:
            raise ValueError(
                f"LTR checkpoint expects {in_features} features; expected {expected_in_features}"
            )
        if hidden_dim <= 0:
            raise ValueError("LTR checkpoint hidden_dim must be positive")

        metadata = checkpoint.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("LTR checkpoint metadata must be a dictionary")

        model = cls(in_features=in_features, hidden_dim=hidden_dim)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(target_device)
        model.eval()
        return model, dict(metadata)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: str | torch.device | None = None,
        expected_in_features: int | None = 7,
    ) -> AdditiveAttentionScorer:
        """Load only the model while preserving the original public interface."""

        model, _ = cls.load_checkpoint(
            path,
            device=device,
            expected_in_features=expected_in_features,
        )
        return model
