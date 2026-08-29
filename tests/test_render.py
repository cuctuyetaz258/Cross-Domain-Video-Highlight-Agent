import json
from pathlib import Path

import pytest

from highlight_agent.media import render
from highlight_agent.schemas import (
    BoundaryAdjustment,
    HighlightCandidate,
    LLMHighlightAssessment,
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


@pytest.mark.parametrize("render_namespace", [None, "openai-gpt-4.1-mini-fixture"])
def test_render_highlights_writes_metadata_without_running_ffmpeg(
    tmp_path: Path,
    monkeypatch,
    render_namespace: str | None,
) -> None:
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

    def fake_thumbnail(source_video, candidate, output_path, **kwargs):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"image")
        return Path(output_path)

    monkeypatch.setattr(render, "render_short_video", fake_render)
    monkeypatch.setattr(render, "render_short_9_16", fake_render)
    monkeypatch.setattr(render, "extract_thumbnail", fake_thumbnail)

    adjustments = [
        BoundaryAdjustment(
            candidate_id=candidate.candidate_id,
            original_start_time=candidate.start_time,
            original_end_time=candidate.end_time,
            start_time=candidate.start_time,
            end_time=candidate.end_time,
            start_source="original",
            end_source="original",
            start_reason="Giữ mốc đề xuất",
            end_reason="Giữ mốc đề xuất",
        )
        for candidate in _candidates()
    ]
    results = render.render_highlights(
        workspace,
        _candidates(),
        transcript=transcript,
        boundary_adjustments=adjustments,
        llm_assessments={
            candidate.candidate_id: LLMHighlightAssessment(
                candidate_id=candidate.candidate_id,
                semantic_relevance=0.8,
                standalone_value=0.8,
                completeness=0.9,
                hook_strength=0.7,
                shareability=0.8,
                title=f"Title {candidate.candidate_id}",
                summary="A transcript-grounded summary.",
                evidence="Hello",
                suggested_start_time=None,
                suggested_end_time=None,
                risk_flags=[],
            )
            for candidate in _candidates()
        },
        render_namespace=render_namespace,
    )

    assert len(results) == 3
    assert results[0].aspect_ratio == "9:16"
    assert results[0].width == 1080
    assert results[0].height == 1920
    result_dir = (
        workspace_dir / "variants" / render_namespace
        if render_namespace
        else workspace_dir
    )
    metadata = json.loads((result_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["video_id"] == "abcdefghijk"
    assert metadata["aspect_ratio"] == "9:16"
    assert len(metadata["highlights"]) == 3
    assert metadata["highlights"][0]["aspect_ratio"] == "9:16"
    assert metadata["boundary_adjustments"][0]["start_source"] == "original"
    assert metadata["highlights"][0]["title"] == "Title candidate-0"
    assert metadata["highlights"][0]["completeness_score"] == 0.9
    assert all(str(result_dir) in str(item.video_path) for item in results)


def test_render_highlights_supports_16_9_landscape(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"fixture")
    workspace_dir = tmp_path / "output" / "landscape_test"
    workspace_dir.mkdir(parents=True)
    workspace = MediaWorkspace(
        video_id="landscape_test",
        source_type="local",
        original_input=str(source),
        source_video_path=source,
        audio_path=workspace_dir / "audio.wav",
        transcript_path=workspace_dir / "transcript.json",
    )
    transcript = TranscriptDocument(
        video_id="landscape_test",
        language="en",
        source="whisper",
        duration=120,
        segments=[TranscriptSegment(id=0, start=1, end=2, text="Hello")],
    )

    monkeypatch.setattr(render, "probe_duration", lambda path: 120.0)

    rendered_kwargs = {}

    def fake_render(source_video, candidate, output_path, **kwargs):
        rendered_kwargs.update(kwargs)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"video")
        return Path(output_path)

    def fake_thumbnail(source_video, candidate, output_path, **kwargs):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"image")
        return Path(output_path)

    monkeypatch.setattr(render, "render_short_video", fake_render)
    monkeypatch.setattr(render, "extract_thumbnail", fake_thumbnail)

    results = render.render_highlights(
        workspace,
        _candidates(),
        aspect_ratio="16:9",
        transcript=transcript,
    )

    assert len(results) == 3
    assert rendered_kwargs.get("aspect_ratio") == "16:9"
    assert results[0].aspect_ratio == "16:9"
    assert results[0].width == 1920
    assert results[0].height == 1080

    metadata = json.loads((workspace_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["aspect_ratio"] == "16:9"
    assert metadata["highlights"][0]["aspect_ratio"] == "16:9"
    assert metadata["highlights"][0]["width"] == 1920
    assert metadata["highlights"][0]["height"] == 1080


def test_get_video_format_spec() -> None:
    spec_916 = render.get_video_format_spec("9:16")
    assert spec_916.width == 1080
    assert spec_916.height == 1920
    assert spec_916.thumbnail_width == 720
    assert spec_916.thumbnail_height == 1280

    spec_169 = render.get_video_format_spec("16:9")
    assert spec_169.width == 1920
    assert spec_169.height == 1080
    assert spec_169.thumbnail_width == 1280
    assert spec_169.thumbnail_height == 720

    import pytest
    with pytest.raises(ValueError, match="unsupported aspect ratio"):
        render.get_video_format_spec("4:3")
