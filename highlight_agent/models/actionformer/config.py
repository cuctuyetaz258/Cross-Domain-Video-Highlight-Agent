from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ActionFormerConfig:
    """Compact ActionFormer configuration for the seven-channel timeline."""

    in_features: int = 7
    input_sample_rate: float = 10.0
    downsample_factor: int = 5
    d_model: int = 128
    num_heads: int = 4
    attention_window: int = 128
    pyramid_levels: int = 4
    blocks_per_level: int = 1
    head_depth: int = 2
    dropout: float = 0.1
    initial_offset_units: float = 30.0
    center_sampling_radius: float = 1.5
    regression_ranges_seconds: tuple[tuple[float, float], ...] = (
        (0.0, 45.0),
        (30.0, 90.0),
        (60.0, 180.0),
        (120.0, float("inf")),
    )
    min_duration_seconds: float = 30.0
    max_duration_seconds: float = 90.0
    score_threshold: float = 0.05
    pre_nms_topk: int = 200

    def __post_init__(self) -> None:
        if self.in_features <= 0 or self.d_model <= 0:
            raise ValueError("in_features and d_model must be positive")
        if self.input_sample_rate <= 0 or self.downsample_factor <= 0:
            raise ValueError("sample rate and downsample factor must be positive")
        if self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if self.attention_window <= 0 or self.pyramid_levels <= 0:
            raise ValueError("attention_window and pyramid_levels must be positive")
        if self.initial_offset_units <= 0:
            raise ValueError("initial_offset_units must be positive")
        if len(self.regression_ranges_seconds) != self.pyramid_levels:
            raise ValueError("one regression range is required per pyramid level")
        if not 0 < self.min_duration_seconds <= self.max_duration_seconds:
            raise ValueError("invalid output duration range")

    @property
    def base_stride_seconds(self) -> float:
        return self.downsample_factor / self.input_sample_rate

    def level_stride_seconds(self, level: int) -> float:
        return self.base_stride_seconds * (2**level)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["regression_ranges_seconds"] = [list(item) for item in self.regression_ranges_seconds]
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ActionFormerConfig:
        values = dict(payload)
        if "regression_ranges_seconds" in values:
            values["regression_ranges_seconds"] = tuple(
                tuple(float(value) for value in item)
                for item in values["regression_ranges_seconds"]
            )
        return cls(**values)
