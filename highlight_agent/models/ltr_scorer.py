from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from highlight_agent.ltr_contract import (
    LTR_CHECKPOINT_VERSION,
    LTRPipelineError,
    validate_feature_contract,
)


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
                "checkpoint_version": LTR_CHECKPOINT_VERSION,
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
        # Checkpoints produced by this project only contain tensors and primitive
        # metadata.  Restrict unpickling so an external artifact cannot execute
        # arbitrary Python globals while being loaded.
        checkpoint = torch.load(path, map_location=target_device, weights_only=True)
        if not isinstance(checkpoint, dict):
            raise ValueError("LTR checkpoint must be a dictionary")

        required = {"state_dict", "in_features", "hidden_dim"}
        missing = sorted(required.difference(checkpoint))
        if missing:
            raise ValueError(f"LTR checkpoint is missing fields: {', '.join(missing)}")

        in_features = int(checkpoint["in_features"])
        hidden_dim = int(checkpoint["hidden_dim"])
        if expected_in_features is not None and in_features != expected_in_features:
            raise ValueError(f"LTR checkpoint expects {in_features} features; expected {expected_in_features}")
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
    def preflight(
        cls,
        path: str | Path | None,
        *,
        device: str | torch.device | None = None,
    ) -> dict[str, Any]:
        """Fail fast and return serializable checkpoint identity/contract data."""

        if not path:
            raise LTRPipelineError(
                "LTR_CHECKPOINT_REQUIRED",
                "provide --ltr-model-path or configure an LTR checkpoint in the UI",
            )
        checkpoint_path = Path(path)
        if not checkpoint_path.is_file():
            raise LTRPipelineError(
                "LTR_CHECKPOINT_NOT_FOUND",
                f"checkpoint does not exist: {checkpoint_path}",
            )
        target_device = torch.device(
            device or ("cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu")
        )
        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location=target_device,
                weights_only=True,
            )
            if not isinstance(checkpoint, dict):
                raise ValueError("checkpoint must be a dictionary")
            version = checkpoint.get("checkpoint_version")
            if version != LTR_CHECKPOINT_VERSION:
                raise LTRPipelineError(
                    "LTR_CHECKPOINT_SCHEMA_MISMATCH",
                    f"checkpoint_version={version!r}; expected {LTR_CHECKPOINT_VERSION!r}",
                )
            model, metadata = cls.load_checkpoint(
                checkpoint_path,
                device=target_device,
                expected_in_features=7,
            )
            del model
            contract = validate_feature_contract(metadata)
        except LTRPipelineError:
            raise
        except Exception as exc:
            raise LTRPipelineError(
                "LTR_CHECKPOINT_LOAD_FAILED",
                f"could not load {checkpoint_path}: {exc}",
            ) from exc

        digest = hashlib.sha256()
        with checkpoint_path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return {
            "path": str(checkpoint_path.resolve()),
            "fingerprint": digest.hexdigest(),
            "checkpoint_version": LTR_CHECKPOINT_VERSION,
            "device": target_device.type,
            "feature_contract": contract,
            "L_ref": float(metadata["L_ref"]),
            "epoch": metadata.get("epoch"),
            "selection_ap": metadata.get("selection_ap"),
            "dataset_fingerprint": metadata["dataset_fingerprint"],
        }

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
