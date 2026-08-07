"""Schema chung cho caption YouTube và transcript Whisper"""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TranscriptWord(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    start: float = Field(ge=0)
    end: float = Field(gt=0)
    text: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_time_range(self) -> Self:
        if self.end <= self.start:
            raise ValueError("word end must be greater than start")
        return self


class TranscriptSegment(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: int = Field(ge=0)
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    text: str = Field(min_length=1)
    words: list[TranscriptWord] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_segment(self) -> Self:
        if self.end <= self.start:
            raise ValueError("segment end must be greater than start")

        previous_start = -1.0
        for word in self.words:
            if word.start < previous_start:
                raise ValueError("segment words must be sorted by start time")
            if word.start < self.start - 0.05 or word.end > self.end + 0.05:
                raise ValueError("word timestamps must stay inside their segment")
            previous_start = word.start
        return self


class Chapter(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    title: str = Field(min_length=1)
    start: float = Field(ge=0)
    end: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_time_range(self) -> Self:
        if self.end <= self.start:
            raise ValueError("chapter end must be greater than start")
        return self


class TranscriptDocument(BaseModel):
    """Transcript chuẩn dùng cho LLM, feature và canh biên"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    schema_version: Literal["1.0"] = "1.0"
    video_id: str = Field(min_length=1)
    language: str = Field(min_length=2)
    source: Literal["youtube_caption", "whisper"]
    duration: float = Field(gt=0)
    segments: list[TranscriptSegment] = Field(min_length=1)
    chapters: list[Chapter] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_timeline(self) -> Self:
        previous_start = -1.0
        for segment in self.segments:
            if segment.start < previous_start:
                raise ValueError("transcript segments must be sorted by start time")
            if segment.end > self.duration + 1.0:
                raise ValueError("segment timestamp exceeds transcript duration")
            previous_start = segment.start

        previous_chapter_start = -1.0
        for chapter in self.chapters:
            if chapter.start < previous_chapter_start:
                raise ValueError("chapters must be sorted by start time")
            if chapter.end > self.duration + 1.0:
                raise ValueError("chapter timestamp exceeds transcript duration")
            previous_chapter_start = chapter.start
        return self
