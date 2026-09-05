"""Content fingerprints and recursive split provenance for nested proposal training."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

NESTED_OOF_VERSION = "actionformer_nested_oof_v2"


def json_digest(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def example_digest(example: Any) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps([example.video_id, example.domain, example.duration]).encode())
    for value in (example.features, example.boundaries, example.importance):
        array = np.ascontiguousarray(value)
        digest.update(json.dumps([array.dtype.str, array.shape]).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def split_contract(train: list[str], val: list[str], test: list[str]) -> dict[str, list[str]]:
    splits = {"train": sorted(train), "val": sorted(val), "test": sorted(test)}
    seen: set[str] = set()
    for name, ids in splits.items():
        if not ids or len(ids) != len(set(ids)):
            raise ValueError(f"{name} split must be non-empty and contain unique video IDs")
        overlap = seen.intersection(ids)
        if overlap:
            raise ValueError(f"split leakage: {sorted(overlap)}")
        seen.update(ids)
    return splits


def lineage_ids(lineage: dict[str, Any]) -> set[str]:
    """Include training AND model-selection data of every upstream checkpoint."""
    required = {"train_video_ids", "selection_video_ids", "ancestors"}
    if not isinstance(lineage, dict) or not required.issubset(lineage):
        raise ValueError("checkpoint has missing data lineage")
    ids: set[str] = set()
    for field in ("train_video_ids", "selection_video_ids"):
        values = lineage[field]
        if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
            raise ValueError(f"invalid lineage {field}")
        if len(values) != len(set(values)):
            raise ValueError(f"duplicate lineage {field}")
        ids.update(values)
    if not isinstance(lineage["ancestors"], list):
        raise ValueError("invalid lineage ancestors")
    for ancestor in lineage["ancestors"]:
        ids.update(lineage_ids(ancestor))
    return ids


def assert_lineage_allowed(lineage: dict[str, Any], allowed: set[str]) -> None:
    unexpected = lineage_ids(lineage) - allowed
    if unexpected:
        raise ValueError(f"upstream training/selection leakage: {sorted(unexpected)}")
