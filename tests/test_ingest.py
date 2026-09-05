import json
from pathlib import Path
from types import SimpleNamespace

from highlight_agent.media import ingest
from highlight_agent.media.ingest import YoutubeMedia
from highlight_agent.schemas import TranscriptDocument, TranscriptSegment


def test_youtube_options_allow_non_mp4_sources_and_configure_cookie_client(tmp_path: Path) -> None:
    options = ingest._youtube_options(tmp_path, "chrome")

    assert "[ext=mp4]" not in options["format"]
    assert options["format_sort"] == ["res:720", "vcodec:h264", "acodec:aac"]
    assert options["cookiesfrombrowser"] == ("chrome",)
    assert options["extractor_args"]["youtube"]["player_client"] == ["default", "web_embedded"]


def test_youtube_options_enable_an_available_javascript_runtime(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        ingest.shutil,
        "which",
        lambda name: "C:/tools/node.exe" if name == "node" else None,
    )

    options = ingest._youtube_options(tmp_path, None)

    assert options["js_runtimes"] == {"node": {"path": "C:/tools/node.exe"}}


def test_download_canonicalizes_shared_youtube_url(tmp_path: Path, monkeypatch) -> None:
    extracted_urls: list[str] = []

    class FakeYoutubeDL:
        def __init__(self, options):
            self.options = options

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def extract_info(self, url, *, download):
            extracted_urls.append(url)
            if download:
                (tmp_path / "source_video.mp4").write_bytes(b"fixture")
            return {"duration": 60, "chapters": []}

    monkeypatch.setattr(
        ingest,
        "yt_dlp",
        SimpleNamespace(
            YoutubeDL=FakeYoutubeDL,
            utils=SimpleNamespace(DownloadError=RuntimeError),
        ),
    )

    media = ingest.download_youtube_media(
        "https://youtu.be/jbL9kl4KPZI?si=tracking&list=playlist",
        tmp_path,
        download_captions=False,
    )

    assert extracted_urls == [
        "https://www.youtube.com/watch?v=jbL9kl4KPZI",
        "https://www.youtube.com/watch?v=jbL9kl4KPZI",
    ]
    assert media.video_path == tmp_path / "source_video.mp4"


def test_youtube_caption_is_used_before_whisper(tmp_path: Path, monkeypatch) -> None:
    output_root = tmp_path / "output"
    caption = tmp_path / "caption.json3"
    caption.write_text(
        json.dumps(
            {
                "events": [
                    {
                        "tStartMs": 0,
                        "dDurationMs": 1000,
                        "segs": [{"utf8": "Caption text"}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    def fake_download(url, workspace_dir, *, cookies_browser=None, download_captions=True):
        assert download_captions is True
        video = Path(workspace_dir) / "source_video.mp4"
        video.write_bytes(b"fixture")
        return YoutubeMedia(video, 60, [], caption, "en")

    def fake_audio(video_path, audio_path):
        Path(audio_path).write_bytes(b"audio")
        return Path(audio_path)

    def fail_whisper(*args, **kwargs):
        raise AssertionError("Whisper must not run when a usable caption exists")

    monkeypatch.setattr(ingest, "download_youtube_media", fake_download)
    monkeypatch.setattr(ingest, "extract_audio_16k_mono", fake_audio)
    monkeypatch.setattr(ingest, "transcribe_with_whisper", fail_whisper)

    workspace = ingest.prepare_media_workspace(
        "https://www.youtube.com/watch?v=jbL9kl4KPZI",
        output_root=output_root,
    )

    assert workspace.has_source_transcript is True
    saved = json.loads(workspace.transcript_path.read_text(encoding="utf-8"))
    assert saved["source"] == "youtube_caption"


def test_whisper_mode_skips_youtube_caption(tmp_path: Path, monkeypatch) -> None:
    def fake_download(url, workspace_dir, *, cookies_browser=None, download_captions=True):
        assert download_captions is False
        video = Path(workspace_dir) / "source_video.mp4"
        video.write_bytes(b"fixture")
        return YoutubeMedia(video, 60, [], None, "en")

    monkeypatch.setattr(ingest, "download_youtube_media", fake_download)
    monkeypatch.setattr(
        ingest,
        "extract_audio_16k_mono",
        lambda video_path, audio_path: Path(audio_path).write_bytes(b"audio") or Path(audio_path),
    )

    def fake_whisper(audio_path, *, video_id, duration, chapters, model_size):
        return TranscriptDocument(
            video_id=video_id,
            language="en",
            source="whisper",
            duration=duration,
            segments=[TranscriptSegment(id=0, start=0, end=1, text="Whisper text")],
        )

    monkeypatch.setattr(ingest, "transcribe_with_whisper", fake_whisper)
    workspace = ingest.prepare_media_workspace(
        "https://www.youtube.com/watch?v=jbL9kl4KPZI",
        output_root=tmp_path / "output",
        transcript_source="whisper",
    )

    saved = json.loads(workspace.transcript_path.read_text(encoding="utf-8"))
    assert workspace.has_source_transcript is False
    assert saved["source"] == "whisper"


def test_youtube_mode_does_not_fallback_to_whisper(tmp_path: Path, monkeypatch) -> None:
    def fake_download(url, workspace_dir, *, cookies_browser=None, download_captions=True):
        video = Path(workspace_dir) / "source_video.mp4"
        video.write_bytes(b"fixture")
        return YoutubeMedia(video, 60, [], None, "en")

    monkeypatch.setattr(ingest, "download_youtube_media", fake_download)

    try:
        ingest.prepare_media_workspace(
            "https://www.youtube.com/watch?v=jbL9kl4KPZI",
            output_root=tmp_path / "output",
            transcript_source="youtube",
        )
    except ingest.MediaProcessingError as exc:
        assert "does not provide" in str(exc)
    else:
        raise AssertionError("YouTube-only mode must fail when caption is unavailable")


def test_local_video_uses_whisper_fallback(tmp_path: Path, monkeypatch) -> None:
    video = tmp_path / "lecture.mp4"
    video.write_bytes(b"fixture")

    monkeypatch.setattr(ingest, "probe_duration", lambda path: 60.0)
    monkeypatch.setattr(
        ingest,
        "extract_audio_16k_mono",
        lambda video_path, audio_path: Path(audio_path).write_bytes(b"audio") or Path(audio_path),
    )

    def fake_whisper(audio_path, *, video_id, duration, chapters, model_size):
        return TranscriptDocument(
            video_id=video_id,
            language="en",
            source="whisper",
            duration=duration,
            segments=[TranscriptSegment(id=0, start=0, end=1, text="Hello")],
        )

    monkeypatch.setattr(ingest, "transcribe_with_whisper", fake_whisper)
    workspace = ingest.prepare_media_workspace(str(video), output_root=tmp_path / "output")

    assert workspace.source_type == "local"
    assert workspace.has_source_transcript is False
    assert workspace.transcript_path.is_file()
