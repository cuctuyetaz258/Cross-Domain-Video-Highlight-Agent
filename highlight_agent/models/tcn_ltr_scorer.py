"""Non-causal temporal-convolution scorer for window-level LTR."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from highlight_agent.features.ltr_contract import (
    LTRPipelineError,
    validate_feature_contract,
)

TCN_LTR_CHECKPOINT_VERSION = "tcn_ltr_v2"


class _ResidualTemporalBlock(nn.Module):
    """One dilated, same-length convolution with a residual connection."""

    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.activation(self.conv(x)))


class TemporalConvLTRScorer(nn.Module):
    """Score a chronological sequence of seven-feature windows with a non-causal TCN."""

    model_type = "tcn_ltr_v2"

    def __init__(
        self,
        in_features: int = 7,
        hidden_dim: int = 32,
        dilations: tuple[int, ...] = (1, 2, 4, 8),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if in_features <= 0 or hidden_dim <= 0 or not dilations:
            raise ValueError("in_features, hidden_dim and dilations must be positive")
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.dilations = tuple(int(value) for value in dilations)
        self.dropout_rate = float(dropout)
        self.proj = nn.Sequential(nn.Linear(in_features, hidden_dim), nn.Tanh())
        self.blocks = nn.ModuleList(
            _ResidualTemporalBlock(hidden_dim, dilation, dropout) for dilation in self.dilations
        )
        self.score_head = nn.Linear(hidden_dim, 1, bias=False)
        self._reset_parameters()

    @property
    def receptive_field_tokens(self) -> int:
        return 1 + 2 * sum(self.dilations)

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a score for every token; accepts ``(W, 7)`` or ``(B, W, 7)``."""

        squeeze_batch = x.ndim == 2
        if squeeze_batch:
            x = x.unsqueeze(0)
        if x.ndim != 3 or x.shape[-1] != self.in_features:
            raise ValueError(f"expected (B, W, {self.in_features}) or (W, {self.in_features}) input")
        encoded = self.proj(x).transpose(1, 2)
        for block in self.blocks:
            encoded = block(encoded)
        scores = self.score_head(encoded.transpose(1, 2)).squeeze(-1)
        return scores.squeeze(0) if squeeze_batch else scores

    def save(self, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "checkpoint_version": TCN_LTR_CHECKPOINT_VERSION,
                "model_type": self.model_type,
                "state_dict": self.state_dict(),
                "in_features": self.in_features,
                "hidden_dim": self.hidden_dim,
                "dilations": self.dilations,
                "dropout": self.dropout_rate,
                "metadata": metadata or {},
            },
            output,
        )

    @classmethod
    def load_checkpoint(
        cls, path: str | Path, *, device: str | torch.device | None = None, expected_in_features: int | None = 7
    ) -> tuple["TemporalConvLTRScorer", dict[str, Any]]:
        target = torch.device(device or "cpu")
        checkpoint = torch.load(path, map_location=target, weights_only=True)
        required = {"state_dict", "in_features", "hidden_dim", "dilations", "dropout", "metadata"}
        if not isinstance(checkpoint, dict) or required.difference(checkpoint):
            raise ValueError("TCN LTR checkpoint is missing required fields")
        if checkpoint.get("checkpoint_version") != TCN_LTR_CHECKPOINT_VERSION:
            raise ValueError("checkpoint is not a TCN LTR v2 checkpoint")
        in_features = int(checkpoint["in_features"])
        if expected_in_features is not None and in_features != expected_in_features:
            raise ValueError(f"checkpoint expects {in_features} features; expected {expected_in_features}")
        metadata = checkpoint["metadata"]
        if not isinstance(metadata, dict):
            raise ValueError("checkpoint metadata must be a dictionary")
        model = cls(
            in_features=in_features,
            hidden_dim=int(checkpoint["hidden_dim"]),
            dilations=tuple(int(value) for value in checkpoint["dilations"]),
            dropout=float(checkpoint["dropout"]),
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.to(target).eval()
        return model, dict(metadata)

    @classmethod
    def preflight(cls, path: str | Path | None, *, device: str | torch.device | None = None) -> dict[str, Any]:
        if not path:
            raise LTRPipelineError("LTR_CHECKPOINT_REQUIRED", "provide a TCN LTR checkpoint")
        checkpoint_path = Path(path)
        if not checkpoint_path.is_file():
            raise LTRPipelineError("LTR_CHECKPOINT_NOT_FOUND", f"checkpoint does not exist: {checkpoint_path}")
        try:
            model, metadata = cls.load_checkpoint(checkpoint_path, device=device)
            contract = validate_feature_contract(metadata)
        except Exception as exc:
            raise LTRPipelineError("LTR_CHECKPOINT_LOAD_FAILED", str(exc)) from exc
        digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        return {
            "path": str(checkpoint_path.resolve()),
            "fingerprint": digest,
            "checkpoint_version": TCN_LTR_CHECKPOINT_VERSION,
            "model_type": cls.model_type,
            "feature_contract": contract,
            "in_features": model.in_features,
            "hidden_dim": model.hidden_dim,
            "dilations": list(model.dilations),
            "receptive_field_tokens": model.receptive_field_tokens,
            "epoch": metadata.get("epoch"),
        }
