"""Validate input, tạo video ID và workspace output"""

import hashlib
import os
import re
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from highlight_agent.schemas import MediaWorkspace

from .errors import InvalidVideoInputError

YOUTUBE_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "youtube-nocookie.com",
    "www.youtube-nocookie.com",
    "youtu.be",
}
YOUTUBE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{11}$")
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".m4v"}


def is_youtube_url(value: str) -> bool:
    try:
        parsed = urlparse(value)
    except (TypeError, ValueError):
        return False
    return parsed.scheme in {"http", "https"} and parsed.hostname in YOUTUBE_HOSTS


def extract_youtube_id(url: str) -> str:
    if not is_youtube_url(url):
        raise InvalidVideoInputError("input is not a valid YouTube URL")

    parsed = urlparse(url)
    if parsed.hostname == "youtu.be":
        candidate = parsed.path.strip("/").split("/")[0]
    elif parsed.path.rstrip("/") == "/watch":
        candidate = parse_qs(parsed.query).get("v", [""])[0]
    else:
        parts = [part for part in parsed.path.split("/") if part]
        candidate = parts[1] if len(parts) >= 2 and parts[0] in {"embed", "shorts", "live"} else ""

    if not YOUTUBE_ID_PATTERN.fullmatch(candidate):
        raise InvalidVideoInputError("YouTube URL does not contain a valid 11-character video ID")
    return candidate


def canonicalize_youtube_url(url: str) -> str:
    """Return one playlist-free watch URL suitable for passing to yt-dlp."""

    return f"https://www.youtube.com/watch?v={extract_youtube_id(url)}"


def local_video_id(path: Path) -> str:
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "-", path.stem).strip("-._") or "video"
    path_hash = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:8]
    return f"{safe_stem}-{path_hash}"


def validate_local_video(video_input: str | os.PathLike[str]) -> Path:
    path = Path(video_input).expanduser().resolve()
    if not path.is_file():
        raise InvalidVideoInputError(f"local video does not exist: {path}")
    if path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_VIDEO_EXTENSIONS))
        raise InvalidVideoInputError(f"unsupported video extension '{path.suffix}'; expected one of: {supported}")
    return path


def create_workspace(
    video_input: str,
    output_root: str | os.PathLike[str] | None = None,
) -> MediaWorkspace:
    if not video_input or not str(video_input).strip():
        raise InvalidVideoInputError("video input must not be empty")

    original_input = str(video_input).strip()
    root = Path(output_root or os.getenv("OUTPUT_DIR", "output")).expanduser()

    if is_youtube_url(original_input):
        source_type = "youtube"
        video_id = extract_youtube_id(original_input)
        source_video_path = root / video_id / "source_video.mp4"
    else:
        source_type = "local"
        source_video_path = validate_local_video(original_input)
        video_id = local_video_id(source_video_path)

    workspace_dir = root / video_id
    (workspace_dir / "shorts").mkdir(parents=True, exist_ok=True)
    (workspace_dir / "thumbnails").mkdir(parents=True, exist_ok=True)

    return MediaWorkspace(
        video_id=video_id,
        source_type=source_type,
        original_input=original_input,
        source_video_path=source_video_path,
        audio_path=workspace_dir / "audio.wav",
        transcript_path=workspace_dir / "transcript.json",
        has_source_transcript=False,
    )
