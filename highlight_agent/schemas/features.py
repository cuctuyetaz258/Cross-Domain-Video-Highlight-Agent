"""Schema cho tín hiệu âm học và tương tác ở Sprint 2"""

import math
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TimeInterval(BaseModel):
    """Một khoảng thời gian không gắn speaker"""

    model_config = ConfigDict(extra="forbid")

    start: float = Field(ge=0)
    end: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_time_range(self) -> Self:
        if self.end <= self.start:
            raise ValueError("interval end must be greater than start")
        return self


class SpeakerTurn(TimeInterval):
    """Một lượt nói đã được diarization gán nhãn"""

    speaker: str = Field(min_length=1)


class AcousticFeatures(BaseModel):
    """Tóm tắt RMS, pitch và silence của toàn bộ audio"""

    model_config = ConfigDict(extra="forbid")

    duration: float = Field(gt=0)
    rms_mean: float = Field(ge=0)
    rms_peak: float = Field(ge=0)
    rms_p95: float = Field(ge=0)
    rms_std: float = Field(ge=0)
    pitch_mean_hz: float | None = Field(default=None, gt=0)
    pitch_median_hz: float | None = Field(default=None, gt=0)
    pitch_std_hz: float | None = Field(default=None, ge=0)
    pitch_min_hz: float | None = Field(default=None, gt=0)
    pitch_max_hz: float | None = Field(default=None, gt=0)
    voiced_ratio: float = Field(ge=0, le=1)
    silence_duration: float = Field(ge=0)
    silence_ratio: float = Field(ge=0, le=1)
    silence_intervals: list[TimeInterval] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_features(self) -> Self:
        numeric_values = (
            self.duration,
            self.rms_mean,
            self.rms_peak,
            self.rms_p95,
            self.rms_std,
            self.voiced_ratio,
            self.silence_duration,
            self.silence_ratio,
            *(
                value
                for value in (
                    self.pitch_mean_hz,
                    self.pitch_median_hz,
                    self.pitch_std_hz,
                    self.pitch_min_hz,
                    self.pitch_max_hz,
                )
                if value is not None
            ),
        )
        if any(not math.isfinite(value) for value in numeric_values):
            raise ValueError("acoustic feature values must be finite")
        if self.silence_duration > self.duration + 1e-6:
            raise ValueError("silence duration cannot exceed audio duration")

        previous_end = 0.0
        for interval in self.silence_intervals:
            if interval.start < previous_end - 1e-6:
                raise ValueError("silence intervals must be sorted and non-overlapping")
            if interval.end > self.duration + 1e-6:
                raise ValueError("silence interval exceeds audio duration")
            previous_end = interval.end
        return self


class InteractionFeatures(BaseModel):
    """Tóm tắt speaker turn và speaker change trong một podcast"""

    model_config = ConfigDict(extra="forbid")

    duration: float = Field(gt=0)
    speaker_count: int = Field(ge=0)
    turn_count: int = Field(ge=0)
    turn_rate_per_minute: float = Field(ge=0)
    speech_duration: float = Field(ge=0)
    speech_ratio: float = Field(ge=0, le=1)
    turns: list[SpeakerTurn] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_features(self) -> Self:
        if self.speech_duration > self.duration + 1e-6:
            raise ValueError("speech duration cannot exceed audio duration")
        if self.speaker_count != len({turn.speaker for turn in self.turns}):
            raise ValueError("speaker_count must match speakers present in turns")
        if any(turn.end > self.duration + 1e-6 for turn in self.turns):
            raise ValueError("speaker turn exceeds audio duration")
        return self


class FeatureWindow(BaseModel):
    """Feature thô của một cửa sổ thời gian dùng cho bộ chấm điểm sau này"""

    model_config = ConfigDict(extra="forbid")

    start: float = Field(ge=0)
    end: float = Field(gt=0)
    acoustic: AcousticFeatures
    interaction: InteractionFeatures | None = None

    @model_validator(mode="after")
    def validate_window(self) -> Self:
        if self.end <= self.start:
            raise ValueError("feature window end must be greater than start")
        if abs(self.acoustic.duration - (self.end - self.start)) > 1e-3:
            raise ValueError("acoustic duration must match feature window duration")
        if self.interaction and abs(self.interaction.duration - (self.end - self.start)) > 1e-3:
            raise ValueError("interaction duration must match feature window duration")
        return self


class FeatureTimeline(BaseModel):
    """Timeline feature thô theo từng video do các extractor Sprint 2 tạo ra"""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    video_id: str = Field(min_length=1)
    domain: Literal["lecture", "podcast", "standup"]
    duration: float = Field(gt=0)
    window_seconds: float = Field(gt=0)
    hop_seconds: float = Field(gt=0)
    acoustic: AcousticFeatures
    interaction: InteractionFeatures | None = None
    windows: list[FeatureWindow] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_timeline(self) -> Self:
        if abs(self.acoustic.duration - self.duration) > 1e-3:
            raise ValueError("global acoustic duration must match timeline duration")
        if self.interaction and abs(self.interaction.duration - self.duration) > 1e-3:
            raise ValueError("global interaction duration must match timeline duration")
        previous_start = -1.0
        for window in self.windows:
            if window.start < previous_start:
                raise ValueError("feature windows must be sorted by start time")
            if window.end > self.duration + 1e-3:
                raise ValueError("feature window exceeds timeline duration")
            previous_start = window.start
        return self
