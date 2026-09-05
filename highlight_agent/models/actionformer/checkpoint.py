from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch

from .config import ActionFormerConfig
from .model import ActionFormerHighlightModel

ACTIONFORMER_CHECKPOINT_VERSION = "3.0"
SUPPORTED_ACTIONFORMER_CHECKPOINT_VERSIONS = {"2.0", ACTIONFORMER_CHECKPOINT_VERSION}
ACTIONFORMER_MODEL_FAMILY = "actionformer_ltr"
EXPECTED_FEATURE_SCHEMA_VERSION = "1.1"
EXPECTED_CHANNEL_ORDER = (
    "rms",
    "pitch",
    "silence",
    "text_score",
    "scene_change",
    "gesture",
    "turn_rate",
)
EXPECTED_INPUT_SAMPLE_RATE = 10.0
EXPECTED_DURATION_RANGE = (30.0, 90.0)


def save_actionformer_checkpoint(
    path: str | Path,
    model: ActionFormerHighlightModel,
    *,
    metadata: dict[str, Any],
    proposal_ltr_state_dict: dict[str, torch.Tensor] | None = None,
    training_state: dict[str, Any] | None = None,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(
        {
            "model_family": ACTIONFORMER_MODEL_FAMILY,
            "checkpoint_version": ACTIONFORMER_CHECKPOINT_VERSION,
            "config": model.config.to_dict(),
            "state_dict": model.state_dict(),
            "proposal_ltr_state_dict": proposal_ltr_state_dict,
            # Optional state is intentionally separate from model metadata: inference
            # consumers can continue to load this checkpoint without knowing its run.
            "training_state": training_state,
            "metadata": metadata,
        },
        temporary,
    )
    temporary.replace(destination)
    return destination


def load_actionformer_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[ActionFormerHighlightModel, dict[str, Any], dict[str, torch.Tensor] | None]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("ActionFormer checkpoint must be a dictionary")
    if checkpoint.get("model_family") != ACTIONFORMER_MODEL_FAMILY:
        raise ValueError("checkpoint model_family is not actionformer_ltr")
    if checkpoint.get("checkpoint_version") not in SUPPORTED_ACTIONFORMER_CHECKPOINT_VERSIONS:
        raise ValueError(f"unsupported ActionFormer checkpoint version: {checkpoint.get('checkpoint_version')!r}")
    config_payload = checkpoint.get("config")
    if not isinstance(config_payload, dict):
        raise ValueError("ActionFormer checkpoint is missing config")
    metadata = checkpoint.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("ActionFormer checkpoint metadata must be a dictionary")
    required_metadata = {
        "feature_schema_version",
        "channel_order",
        "dataset_fingerprint",
        "split_fingerprint",
        "normalization_policy_version",
    }
    missing = sorted(required_metadata.difference(metadata))
    if missing:
        raise ValueError("ActionFormer checkpoint metadata is missing: " + ", ".join(missing))
    config = ActionFormerConfig.from_dict(config_payload)
    if metadata["feature_schema_version"] != EXPECTED_FEATURE_SCHEMA_VERSION:
        raise ValueError("ActionFormer checkpoint feature schema is incompatible")
    if tuple(metadata["channel_order"]) != EXPECTED_CHANNEL_ORDER:
        raise ValueError("ActionFormer checkpoint channel order is incompatible")
    if config.in_features != len(EXPECTED_CHANNEL_ORDER):
        raise ValueError("ActionFormer checkpoint input channel count is incompatible")
    if config.input_sample_rate != EXPECTED_INPUT_SAMPLE_RATE:
        raise ValueError("ActionFormer checkpoint input sample rate is incompatible")
    if (config.min_duration_seconds, config.max_duration_seconds) != EXPECTED_DURATION_RANGE:
        raise ValueError("ActionFormer checkpoint duration range is incompatible")
    model = ActionFormerHighlightModel(config)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, metadata, checkpoint.get("proposal_ltr_state_dict")


def actionformer_checkpoint_info(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    model, metadata, proposal_ltr = load_actionformer_checkpoint(source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    return {
        "path": str(source.resolve()),
        "fingerprint": digest,
        "model_family": ACTIONFORMER_MODEL_FAMILY,
        "checkpoint_version": ACTIONFORMER_CHECKPOINT_VERSION,
        "config": model.config.to_dict(),
        "metadata": metadata,
        "has_proposal_ltr": proposal_ltr is not None,
    }
