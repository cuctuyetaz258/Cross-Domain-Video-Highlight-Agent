"""Chuẩn hóa YouTube và file local về cùng MediaWorkspace"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yt_dlp
from dotenv import load_dotenv

from highlight_agent.schemas import Chapter, MediaWorkspace, TranscriptDocument

from .audio import extract_audio_16k_mono, probe_duration
from .errors import MediaProcessingError
from .transcript import parse_youtube_json3, save_transcript, transcribe_with_whisper
from .workspace import create_workspace


@dataclass(frozen=True)
class YoutubeMedia:
    video_path: Path
    duration: float
    chapters: list[Chapter]
    caption_path: Path | None
    caption_language: str = "en"


def _preferred_english_track(info: dict[str, Any]) -> tuple[str, str] | None:
    for source_name in ("subtitles", "automatic_captions"):
        tracks = info.get(source_name) or {}
        keys = list(tracks)
        ordered_keys = [key for key in ("en", "en-orig", "en-US", "en-GB") if key in tracks]
        ordered_keys.extend(key for key in keys if key.lower().startswith("en") and key not in ordered_keys)
        if ordered_keys:
            return source_name, ordered_keys[0]
    return None


def _youtube_options(workspace_dir: Path, cookies_browser: str | None) -> dict[str, Any]:
    options: dict[str, Any] = {
        "format": (
            "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/"
            "best[height<=720][ext=mp4]/bestvideo[height<=720]+bestaudio/best[height<=720]"
        ),
        "outtmpl": str(workspace_dir / "source_video.%(ext)s"),
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": False,
        "sleep_requests": 1.5,
    }
    try:
        from yt_dlp.networking.impersonate import ImpersonateTarget
        options["impersonate"] = ImpersonateTarget(client="chrome")
    except ImportError:
        pass # Older yt-dlp versions might not have this
    if cookies_browser:
        options["cookiesfrombrowser"] = (cookies_browser,)
    return options


def _chapters_from_info(info: dict[str, Any], duration: float) -> list[Chapter]:
    chapters: list[Chapter] = []
    for raw_chapter in info.get("chapters") or []:
        start = float(raw_chapter.get("start_time") or 0)
        end = float(raw_chapter.get("end_time") or duration)
        title = str(raw_chapter.get("title") or "Untitled chapter").strip()
        if end > start:
            chapters.append(Chapter(title=title, start=start, end=min(end, duration)))
    return chapters


def _find_downloaded_video(workspace_dir: Path) -> Path:
    preferred = workspace_dir / "source_video.mp4"
    if preferred.is_file():
        return preferred
    candidates = [
        path
        for path in workspace_dir.glob("source_video.*")
        if path.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov", ".m4v"}
    ]
    if not candidates:
        raise MediaProcessingError("yt-dlp completed without producing a local video file")
    return candidates[0]


def _find_caption_file(workspace_dir: Path, language: str) -> Path | None:
    exact = workspace_dir / f"source_video.{language}.json3"
    if exact.is_file():
        return exact
    candidates = sorted(workspace_dir.glob("source_video*.json3"))
    return candidates[0] if candidates else None


def download_youtube_media(
    url: str,
    workspace_dir: str | Path,
    *,
    cookies_browser: str | None = None,
    download_captions: bool = True,
) -> YoutubeMedia:
    workspace_path = Path(workspace_dir)
    
    # 1. EARLY EXIT CACHING
    try:
        cached_video = _find_downloaded_video(workspace_path)
        if cached_video.stat().st_size > 0:
            duration = probe_duration(cached_video)
            cached_caption = _find_caption_file(workspace_path, "en") if download_captions else None
            # Trả về luôn nếu đã có file cache
            return YoutubeMedia(cached_video, duration, [], cached_caption, "en")
    except MediaProcessingError:
        pass # Không tìm thấy cache, tiếp tục tải

    # 2. SINGLE-PASS DOWNLOAD
    options = _youtube_options(workspace_path, cookies_browser)
    if download_captions:
        options.update({
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": ["en.*", "en"], # Lấy tiếng Anh
            "subtitlesformat": "json3",
        })

    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=True)
    except yt_dlp.utils.DownloadError as exc:
        raise MediaProcessingError(f"YouTube download failed: {exc}") from exc

    duration = float(info.get("duration") or 0)
    video_path = _find_downloaded_video(workspace_path)
    if duration <= 0:
        duration = probe_duration(video_path)
    chapters = _chapters_from_info(info, duration)
    
    caption_path = _find_caption_file(workspace_path, "en") if download_captions else None
    language = "en" # Định dạng lang mặc định sau khi lưu
    
    return YoutubeMedia(video_path, duration, chapters, caption_path, language)


def prepare_media_workspace(
    video_input: str,
    *,
    output_root: str | Path | None = None,
    cookies_browser: str | None = None,
    whisper_model_size: str = "base.en",
    transcript_source: Literal["auto", "youtube", "whisper"] = "auto",
) -> MediaWorkspace:
    """Chuẩn bị video, audio và transcript theo caption-first"""

    if transcript_source not in {"auto", "youtube", "whisper"}:
        raise ValueError("transcript_source must be 'auto', 'youtube', or 'whisper'")

    load_dotenv()
    workspace = create_workspace(video_input, output_root)
    workspace_dir = workspace.transcript_path.parent
    transcript: TranscriptDocument | None = None

    if workspace.source_type == "youtube":
        selected_browser = cookies_browser
        if selected_browser is None:
            selected_browser = os.getenv("YTDLP_COOKIES_BROWSER") or None
        youtube = download_youtube_media(
            workspace.original_input,
            workspace_dir,
            cookies_browser=selected_browser,
            download_captions=transcript_source != "whisper",
        )
        workspace = workspace.model_copy(update={"source_video_path": youtube.video_path})
        duration = youtube.duration
        chapters = youtube.chapters
        if youtube.caption_path and transcript_source != "whisper":
            try:
                transcript = parse_youtube_json3(
                    youtube.caption_path,
                    video_id=workspace.video_id,
                    duration=duration,
                    chapters=chapters,
                    language=youtube.caption_language,
                )
            except MediaProcessingError as exc:
                if transcript_source == "youtube":
                    raise MediaProcessingError("YouTube caption could not be parsed") from exc
                transcript = None
        elif transcript_source == "youtube":
            raise MediaProcessingError("YouTube video does not provide a usable English caption")
    else:
        if transcript_source == "youtube":
            raise MediaProcessingError("YouTube transcript source cannot be used with a local video")
        duration = probe_duration(workspace.source_video_path)
        chapters = []

    extract_audio_16k_mono(workspace.source_video_path, workspace.audio_path)

    if transcript is None:
        transcript = transcribe_with_whisper(
            workspace.audio_path,
            video_id=workspace.video_id,
            duration=duration,
            chapters=chapters,
            model_size=whisper_model_size,
        )

    save_transcript(transcript, workspace.transcript_path)
    return workspace.model_copy(update={"has_source_transcript": transcript.source == "youtube_caption"})
