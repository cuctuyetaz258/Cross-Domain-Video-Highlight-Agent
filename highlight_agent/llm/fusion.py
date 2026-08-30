"""Percentile-rank normalization and portable LTR–LLM fusion artifacts."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

FUSION_SCHEMA_VERSION = "1.0"
FUSION_METHOD = "percentile_rank_global_alpha"


def percentile_rank(values: Iterable[float]) -> np.ndarray:
    """Map values to average percentile ranks in [0, 1], preserving ties."""

    array = np.asarray(list(values), dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("percentile_rank requires a non-empty one-dimensional sequence")
    if not np.isfinite(array).all():
        raise ValueError("percentile_rank values must be finite")
    if array.size == 1 or float(array.max() - array.min()) <= 1e-12:
        return np.full(array.shape, 0.5, dtype=np.float64)

    ranks = np.empty(array.size, dtype=np.float64)
    for value in np.unique(array):
        indices = np.flatnonzero(array == value)
        lower_count = int(np.count_nonzero(array < value))
        upper_position = lower_count + len(indices) - 1
        average_position = (lower_count + upper_position) / 2.0
        ranks[indices] = average_position / (array.size - 1)
    return ranks


def fuse_ranked_scores(
    ltr_scores: Iterable[float],
    llm_scores: Iterable[float],
    *,
    alpha: float,
) -> np.ndarray:
    """Rank-normalize each score source and combine them with one parameter."""

    if not 0 <= alpha <= 1 or not math.isfinite(alpha):
        raise ValueError("alpha must be finite and between 0 and 1")
    ltr = percentile_rank(ltr_scores)
    llm = percentile_rank(llm_scores)
    if ltr.shape != llm.shape:
        raise ValueError("LTR and LLM score arrays must have equal length")
    return alpha * ltr + (1.0 - alpha) * llm


@dataclass(frozen=True)
class FusionCalibrator:
    """Validated fusion configuration loaded independently from the LTR model."""

    alpha: float
    selection_metric: str
    selection_score: float | None = None
    ltr_checkpoint_fingerprint: str | None = None
    llm_model: str | None = None
    prompt_version: str | None = None
    training_dataset_fingerprint: str | None = None
    training_checkpoint_fingerprints: tuple[str, ...] = ()
    schema_version: str = FUSION_SCHEMA_VERSION
    method: str = FUSION_METHOD
    source_path: str | None = None

    def __post_init__(self) -> None:
        if self.schema_version != FUSION_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported fusion schema {self.schema_version!r}; expected {FUSION_SCHEMA_VERSION!r}"
            )
        if self.method != FUSION_METHOD:
            raise ValueError(f"unsupported fusion method: {self.method!r}")
        if not 0 <= self.alpha <= 1 or not math.isfinite(self.alpha):
            raise ValueError("fusion alpha must be finite and between 0 and 1")
        if not self.selection_metric:
            raise ValueError("fusion selection_metric is required")

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_checkpoint_fingerprint: str | None = None,
        expected_llm_model: str | None = None,
        expected_prompt_version: str | None = None,
    ) -> "FusionCalibrator":
        source = Path(path)
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("fusion calibrator must be a JSON object")
        alpha = payload.get("alpha", payload.get("ltr_weight"))
        calibrator = cls(
            alpha=float(alpha),
            selection_metric=str(payload.get("selection_metric", "")),
            selection_score=(
                float(payload["selection_score"])
                if payload.get("selection_score") is not None
                else None
            ),
            ltr_checkpoint_fingerprint=payload.get("ltr_checkpoint_fingerprint"),
            llm_model=payload.get("llm_model"),
            prompt_version=payload.get("prompt_version"),
            training_dataset_fingerprint=payload.get("training_dataset_fingerprint"),
            training_checkpoint_fingerprints=tuple(
                str(value)
                for value in payload.get("training_checkpoint_fingerprints", [])
            ),
            schema_version=str(payload.get("schema_version", "")),
            method=str(payload.get("method", "")),
            source_path=str(source.resolve()),
        )
        expected = {
            "ltr_checkpoint_fingerprint": expected_checkpoint_fingerprint,
            "llm_model": expected_llm_model,
            "prompt_version": expected_prompt_version,
        }
        for field, expected_value in expected.items():
            actual = getattr(calibrator, field)
            if expected_value and actual != expected_value:
                raise ValueError(
                    f"fusion calibrator {field}={actual!r}, expected {expected_value!r}"
                )
        return calibrator

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "method": self.method,
            "normalization": "per_video_average_percentile_rank",
            "alpha": self.alpha,
            "ltr_weight": self.alpha,
            "llm_weight": 1.0 - self.alpha,
            "selection_metric": self.selection_metric,
            "selection_score": self.selection_score,
            "ltr_checkpoint_fingerprint": self.ltr_checkpoint_fingerprint,
            "llm_model": self.llm_model,
            "prompt_version": self.prompt_version,
            "training_dataset_fingerprint": self.training_dataset_fingerprint,
            "training_checkpoint_fingerprints": list(
                self.training_checkpoint_fingerprints
            ),
        }
