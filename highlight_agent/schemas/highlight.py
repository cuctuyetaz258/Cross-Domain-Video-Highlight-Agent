"""Schema cho candidate và clip highlight đã render"""

import math
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class HighlightCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    candidate_id: str = Field(min_length=1)
    start_time: float = Field(ge=0)
    end_time: float = Field(gt=0)
    score: float = Field(ge=0)
    reason: str = Field(min_length=1)
    signals: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_candidate(self) -> Self:
        duration = self.end_time - self.start_time
        if duration < 30 or duration > 90:
            raise ValueError("highlight duration must be between 30 and 90 seconds")
        if not math.isfinite(self.score):
            raise ValueError("highlight score must be finite")
        if any(not math.isfinite(value) for value in self.signals.values()):
            raise ValueError("all signal values must be finite")
        return self


class BoundaryAdjustment(BaseModel):
    """Lưu mốc đề xuất, mốc đã canh biên và lý do thay đổi"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    candidate_id: str = Field(min_length=1)
    original_start_time: float = Field(ge=0)
    original_end_time: float = Field(gt=0)
    start_time: float = Field(ge=0)
    end_time: float = Field(gt=0)
    start_source: Literal["punctuation", "silence", "segment_fallback", "original"]
    end_source: Literal["punctuation", "silence", "segment_fallback", "original"]
    start_reason: str = Field(min_length=1)
    end_reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_time_ranges(self) -> Self:
        if self.original_end_time <= self.original_start_time:
            raise ValueError("original boundary end must be greater than start")
        if self.end_time <= self.start_time:
            raise ValueError("refined boundary end must be greater than start")
        return self


class RenderedHighlight(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    candidate_id: str = Field(min_length=1)
    video_path: Path
    thumbnail_path: Path | None = None
    start_time: float = Field(ge=0)
    end_time: float = Field(gt=0)
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_time_range(self) -> Self:
        duration = self.end_time - self.start_time
        if duration < 30 or duration > 90:
            raise ValueError("rendered highlight duration must be between 30 and 90 seconds")
        return self
