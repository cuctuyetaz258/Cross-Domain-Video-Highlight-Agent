"""Xử lý input, audio, transcript, render và thumbnail"""

from .audio import extract_audio_16k_mono, probe_duration
from .errors import InvalidVideoInputError, MediaProcessingError
from .ingest import download_youtube_media, prepare_media_workspace
from .render import (
    DEFAULT_ASPECT_RATIO,
    VIDEO_FORMAT_SPECS,
    AspectRatio,
    VideoFormatSpec,
    extract_thumbnail,
    get_video_format_spec,
    render_highlights,
    render_short_9_16,
    render_short_video,
    write_highlight_srt,
)
from .transcript import parse_youtube_json3, save_transcript, transcribe_with_whisper
from .workspace import create_workspace, extract_youtube_id, is_youtube_url

__all__ = [
    "DEFAULT_ASPECT_RATIO",
    "VIDEO_FORMAT_SPECS",
    "AspectRatio",
    "InvalidVideoInputError",
    "MediaProcessingError",
    "VideoFormatSpec",
    "create_workspace",
    "download_youtube_media",
    "extract_audio_16k_mono",
    "extract_thumbnail",
    "extract_youtube_id",
    "get_video_format_spec",
    "is_youtube_url",
    "parse_youtube_json3",
    "prepare_media_workspace",
    "probe_duration",
    "render_highlights",
    "render_short_9_16",
    "render_short_video",
    "save_transcript",
    "transcribe_with_whisper",
    "write_highlight_srt",
]
