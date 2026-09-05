"""Canonical feature and checkpoint contract for the required LTR pipeline."""

from __future__ import annotations

from typing import Any

from highlight_agent.ltr_contract import LTR_CHANNEL_ORDER, LTR_FEATURE_SCHEMA_VERSION

LTR_CHECKPOINT_VERSION = "1.0"
LTR_SAMPLE_RATE = 10
LTR_NORMALIZATION = "minmax_per_video"
LTR_WINDOW_SIZE = 50
LTR_HOP_SIZE = 10
LTR_OUTPUT_CLIP_SECONDS = 30.0
LTR_EXTRACTOR_VERSION = "1.1"


class LTRPipelineError(RuntimeError):
    """A structured, user-actionable failure in the required LTR path."""

    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


def feature_contract() -> dict[str, Any]:
    """Return the exact feature contract embedded in new checkpoints."""

    return {
        "schema_version": LTR_FEATURE_SCHEMA_VERSION,
        "channel_order": list(LTR_CHANNEL_ORDER),
        "sample_rate": LTR_SAMPLE_RATE,
        "normalization": LTR_NORMALIZATION,
        "window_size": LTR_WINDOW_SIZE,
        "hop_size": LTR_HOP_SIZE,
        "window_sec": LTR_WINDOW_SIZE / LTR_SAMPLE_RATE,
        "hop_sec": LTR_HOP_SIZE / LTR_SAMPLE_RATE,
        "output_clip_seconds": LTR_OUTPUT_CLIP_SECONDS,
        "extractor_version": LTR_EXTRACTOR_VERSION,
    }


def validate_feature_contract(metadata: dict[str, Any]) -> dict[str, Any]:
    """Validate checkpoint metadata against the production feature contract."""

    schema = metadata.get("feature_schema")
    if not isinstance(schema, dict):
        raise LTRPipelineError(
            "LTR_CHECKPOINT_SCHEMA_MISMATCH",
            "checkpoint metadata is missing feature_schema; rebuild caches and retrain",
        )

    expected = feature_contract()
    mismatches = [
        f"{key}={schema.get(key)!r} (expected {value!r})" for key, value in expected.items() if schema.get(key) != value
    ]
    if metadata.get("schema_version") != LTR_FEATURE_SCHEMA_VERSION:
        mismatches.insert(
            0,
            f"schema_version={metadata.get('schema_version')!r} (expected {LTR_FEATURE_SCHEMA_VERSION!r})",
        )
    if mismatches:
        raise LTRPipelineError("LTR_CHECKPOINT_SCHEMA_MISMATCH", "; ".join(mismatches))

    l_ref = metadata.get("L_ref")
    if not isinstance(l_ref, (int, float)) or float(l_ref) <= 0:
        raise LTRPipelineError(
            "LTR_CHECKPOINT_SCHEMA_MISMATCH",
            "checkpoint metadata L_ref must be a positive number",
        )
    if not metadata.get("dataset_fingerprint"):
        raise LTRPipelineError(
            "LTR_CHECKPOINT_SCHEMA_MISMATCH",
            "checkpoint metadata is missing dataset_fingerprint",
        )
    return expected
