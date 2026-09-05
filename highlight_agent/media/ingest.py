"""Chuẩn hóa YouTube và file local về cùng MediaWorkspace"""

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

try:
    import yt_dlp
except ImportError:
    yt_dlp = None
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

from highlight_agent.schemas import Chapter, MediaWorkspace, TranscriptDocument

from .audio import extract_audio_16k_mono, probe_duration
from .errors import MediaProcessingError
from .transcript import parse_youtube_json3, save_transcript, transcribe_with_whisper
from .workspace import canonicalize_youtube_url, create_workspace


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
        # Do not require MP4/M4A inputs. YouTube may expose only WebM, HLS, or
        # combined formats for a given player client; ffmpeg can still merge or
        # remux the selected streams to the requested MP4 output.
        "format": "bv*[height<=720]+ba/b[height<=720]/b",
        # Prefer broadly decodable codecs for OpenCV and rendered MP4 output,
        # while retaining the generic selector as a last-resort fallback.
        "format_sort": ["res:720", "vcodec:h264", "acodec:aac"],
        "outtmpl": str(workspace_dir / "source_video.%(ext)s"),
        "merge_output_format": "mp4",
        "noplaylist": True,
        "retries": 3,
        "fragment_retries": 3,
        "quiet": True,
        "no_warnings": False,
    }
    for runtime, executable_name in (("deno", "deno"), ("node", "node"), ("quickjs", "qjs")):
        executable = shutil.which(executable_name)
        if executable:
            options["js_runtimes"] = {runtime: {"path": executable}}
            break
    if cookies_browser:
        options["cookiesfrombrowser"] = (cookies_browser,)
        # Logged-in extraction currently defaults to a YouTube client that can
        # return "The page needs to be reloaded". Keep the normal clients and
        # add web_embedded as the documented compatibility fallback.
        options["extractor_args"] = {
            "youtube": {"player_client": ["default", "web_embedded"]}
        }
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
    if yt_dlp is None:
        raise MediaProcessingError("yt-dlp is not installed")
    workspace_path = Path(workspace_dir)
    workspace_path.mkdir(parents=True, exist_ok=True)
    canonical_url = canonicalize_youtube_url(url)
    options = _youtube_options(workspace_path, cookies_browser)

    try:
        with yt_dlp.YoutubeDL({**options, "skip_download": True}) as ydl:
            initial_info = ydl.extract_info(canonical_url, download=False)

        track = _preferred_english_track(initial_info) if download_captions else None
        download_options = dict(options)
        if track:
            _, language = track
            download_options.update(
                {
                    "writesubtitles": True,
                    "writeautomaticsub": True,
                    "subtitleslangs": [language],
                    "subtitlesformat": "json3",
                }
            )

        with yt_dlp.YoutubeDL(download_options) as ydl:
            info = ydl.extract_info(canonical_url, download=True)
    except yt_dlp.utils.DownloadError as exc:
        message = str(exc)
        if "not installed" in message.lower():
            hint = "Install project dependencies so the yt-dlp package is available."
        elif "sign in" in message.lower() or "not a bot" in message.lower():
            hint = "Update yt-dlp, then configure YTDLP_COOKIES_BROWSER with a closed, logged-in browser."
        elif "page needs to be reloaded" in message.lower():
            hint = "Update yt-dlp; the project already enables the web_embedded client fallback when cookies are used."
        elif "requested format" in message.lower() or "no video formats" in message.lower():
            hint = "Update yt-dlp and verify that ffmpeg is installed; YouTube did not expose a compatible media format."
        else:
            hint = "Update yt-dlp and retry; use browser cookies only when YouTube requires authentication."
        raise MediaProcessingError(f"YouTube download failed: {message}. {hint}") from exc

    duration = float(info.get("duration") or initial_info.get("duration") or 0)
    video_path = _find_downloaded_video(workspace_path)
    if duration <= 0:
        duration = probe_duration(video_path)
    chapters = _chapters_from_info(info, duration)
    language = track[1] if track else "en"
    caption_path = _find_caption_file(workspace_path, language) if track else None
    return YoutubeMedia(video_path, duration, chapters, caption_path, language)


def prepare_media_workspace(
    video_input: str,
    *,
    output_root: str | Path | None = None,
    cookies_browser: str | None = None,
    whisper_model_size: str = "small.en",
    transcript_source: Literal["auto", "youtube", "whisper"] = "auto",
) -> MediaWorkspace:
    """Chuẩn bị video, audio và transcript theo caption-first"""

    if transcript_source not in {"auto", "youtube", "whisper"}:
        raise ValueError("transcript_source must be 'auto', 'youtube', or 'whisper'")

    if load_dotenv is not None:
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
