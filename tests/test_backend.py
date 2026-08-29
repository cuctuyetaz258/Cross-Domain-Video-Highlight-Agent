from pathlib import Path

from highlight_agent import backend


def test_prepare_video_forces_whisper_transcripts(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_prepare_media_workspace(video_input: str, **kwargs):
        captured["video_input"] = video_input
        captured.update(kwargs)
        return "workspace"

    monkeypatch.setattr(backend, "prepare_media_workspace", fake_prepare_media_workspace)

    workspace = backend.prepare_video(
        "https://www.youtube.com/watch?v=abcdefghijk",
        output_root=tmp_path,
        cookies_browser="chrome",
        transcript_source="youtube",
    )

    assert workspace == "workspace"
    assert captured == {
        "video_input": "https://www.youtube.com/watch?v=abcdefghijk",
        "output_root": tmp_path,
        "cookies_browser": "chrome",
        "transcript_source": "whisper",
    }
