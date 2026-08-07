"""Hàm hỗ trợ ffmpeg, ffprobe và audio 16 kHz mono"""

import json
import shutil
import subprocess
from pathlib import Path

from .errors import MediaProcessingError


def require_executable(name: str) -> str:
    executable = shutil.which(name)
    if not executable:
        raise MediaProcessingError(f"required executable '{name}' was not found in PATH")
    return executable


def run_media_command(command: list[str], operation: str) -> None:
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or str(exc)).strip()
        raise MediaProcessingError(f"{operation} failed: {detail}") from exc


def probe_duration(media_path: str | Path) -> float:
    ffprobe = require_executable("ffprobe")
    command = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(media_path),
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        duration = float(json.loads(result.stdout)["format"]["duration"])
    except (subprocess.CalledProcessError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise MediaProcessingError(f"could not read duration for {media_path}") from exc
    if duration <= 0:
        raise MediaProcessingError(f"media duration must be positive: {media_path}")
    return duration


def extract_audio_16k_mono(video_path: str | Path, audio_path: str | Path) -> Path:
    ffmpeg = require_executable("ffmpeg")
    output = Path(audio_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(output),
    ]
    run_media_command(command, "audio extraction")
    return output
