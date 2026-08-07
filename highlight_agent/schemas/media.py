"""Schema cho input media và workspace xử lý"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator


class MediaWorkspace(BaseModel):
    """Output chung cho video YouTube và file local"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    schema_version: Literal["1.0"] = "1.0"
    video_id: str
    source_type: Literal["youtube", "local"]
    original_input: str
    source_video_path: Path
    audio_path: Path
    transcript_path: Path
    has_source_transcript: bool = False

    @field_validator("video_id")
    @classmethod
    def validate_video_id(cls, value: str) -> str:
        if not value:
            raise ValueError("video_id must not be empty")
        if "/" in value or "\\" in value or value in {".", ".."}:
            raise ValueError("video_id must not contain path separators")
        return value
