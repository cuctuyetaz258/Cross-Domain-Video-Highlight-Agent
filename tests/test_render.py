import json
from pathlib import Path

from highlight_agent.media import render
from highlight_agent.schemas import (
    HighlightCandidate,
    MediaWorkspace,
    TranscriptDocument,
    TranscriptSegment,
)


def _candidates() -> list[HighlightCandidate]:
    return [
        HighlightCandidate(
            candidate_id=f"candidate-{index}",
            start_time=index * 30,
            end_time=index * 30 + 30,
            score=5 - index,
            reason="Test highlight",
        )
        for index in range(3)
    ]


def test_write_highlight_srt_uses_relative_timestamps(tmp_path: Path) -> None:
    transcript = TranscriptDocument(
        video_id="abcdefghijk",
        language="en",
        source="whisper",
        duration=120,
        segments=[TranscriptSegment(id=0, start=31, end=33, text="Hello world")],
    )
    path = render.write_highlight_srt(transcript, _candidates()[1], tmp_path / "clip.srt")

    assert path is not None
    content = path.read_text(encoding="utf-8")
    assert "00:00:01,000 --> 00:00:03,000" in content


def test_write_highlight_srt_removes_rolling_caption_overlap(tmp_path: Path) -> None:
    transcript = TranscriptDocument(
        video_id="abcdefghijk",
        language="en",
        source="youtube_caption",
        duration=90,
        segments=[
            TranscriptSegment(id=0, start=28, end=32, text="Old caption"),
            TranscriptSegment(id=1, start=30, end=35, text="Current caption"),
            TranscriptSegment(id=2, start=33, end=38, text="Next caption"),
        ],
    )
    path = render.write_highlight_srt(transcript, _candidates()[1], tmp_path / "clip.srt")

    assert path is not None
    content = path.read_text(encoding="utf-8")
    assert "Old caption" not in content
    assert "00:00:00,000 --> 00:00:03,000\nCurrent caption" in content
    assert "00:00:03,000 --> 00:00:08,000\nNext caption" in content


def test_render_highlights_writes_metadata_without_running_ffmpeg(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"fixture")
    workspace_dir = tmp_path / "output" / "abcdefghijk"
    workspace_dir.mkdir(parents=True)
    workspace = MediaWorkspace(
        video_id="abcdefghijk",
        source_type="local",
        original_input=str(source),
        source_video_path=source,
        audio_path=workspace_dir / "audio.wav",
        transcript_path=workspace_dir / "transcript.json",
    )
    transcript = TranscriptDocument(
        video_id="abcdefghijk",
        language="en",
        source="whisper",
        duration=120,
        segments=[TranscriptSegment(id=0, start=1, end=2, text="Hello")],
    )

    monkeypatch.setattr(render, "probe_duration", lambda path: 120.0)

    def fake_render(source_video, candidate, output_path, **kwargs):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"video")
        return Path(output_path)

    def fake_thumbnail(source_video, candidate, output_path):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"image")
        return Path(output_path)

    monkeypatch.setattr(render, "render_short_9_16", fake_render)
    monkeypatch.setattr(render, "extract_thumbnail", fake_thumbnail)

    results = render.render_highlights(workspace, _candidates(), transcript=transcript)

    assert len(results) == 3
    metadata = json.loads((workspace_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["video_id"] == "abcdefghijk"
    assert len(metadata["highlights"]) == 3
