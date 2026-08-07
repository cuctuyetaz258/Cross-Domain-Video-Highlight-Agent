from pathlib import Path

from highlight_agent.agent import build_agent_graph, nodes
from highlight_agent.schemas import (
    MediaWorkspace,
    RenderedHighlight,
    TranscriptDocument,
    TranscriptSegment,
)


def _workspace(tmp_path: Path) -> MediaWorkspace:
    workspace_dir = tmp_path / "output" / "abcdefghijk"
    workspace_dir.mkdir(parents=True)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"fixture")
    return MediaWorkspace(
        video_id="abcdefghijk",
        source_type="local",
        original_input=str(source),
        source_video_path=source,
        audio_path=workspace_dir / "audio.wav",
        transcript_path=workspace_dir / "transcript.json",
    )


def _transcript() -> TranscriptDocument:
    return TranscriptDocument(
        video_id="abcdefghijk",
        language="en",
        source="whisper",
        duration=360,
        segments=[TranscriptSegment(id=0, start=0, end=2, text="Hello world")],
    )


def test_plan_uses_domain_specific_profile() -> None:
    lecture = nodes.plan({"video_path": "video.mp4", "domain": "lecture"})["profile"]
    podcast = nodes.plan({"video_path": "video.mp4", "domain": "podcast"})["profile"]

    assert lecture["linguistic"] == 0.50
    assert podcast["interaction"] == 0.30
    assert sum(lecture.values()) == 1


def test_analyze_naive_baseline_is_reproducible(tmp_path: Path) -> None:
    state = {
        "video_path": "video.mp4",
        "domain": "lecture",
        "workspace": _workspace(tmp_path),
        "transcript": _transcript(),
        "profile": nodes.PROFILE_WEIGHTS["lecture"],
    }

    first = nodes.analyze(state)
    second = nodes.analyze(state)

    assert first["candidates"] == second["candidates"]
    assert len(first["candidates"]) == 5
    assert all(candidate.end_time <= 360 for candidate in first["candidates"])


def test_full_graph_calls_backend_facade(tmp_path: Path, monkeypatch) -> None:
    workspace = _workspace(tmp_path)
    transcript = _transcript()
    workspace.transcript_path.write_text(transcript.model_dump_json(), encoding="utf-8")

    calls = {}

    def fake_prepare(*args, **kwargs):
        calls["transcript_source"] = kwargs["transcript_source"]
        return workspace

    monkeypatch.setattr(nodes, "prepare_video", fake_prepare)
    monkeypatch.setattr(nodes, "load_transcript", lambda path: transcript)

    def fake_render(workspace_arg, candidates, *, burn_subtitles=True):
        calls["burn_subtitles"] = burn_subtitles
        return [
            RenderedHighlight(
                candidate_id=candidate.candidate_id,
                video_path=workspace.transcript_path.parent / f"{candidate.candidate_id}.mp4",
                thumbnail_path=None,
                start_time=candidate.start_time,
                end_time=candidate.end_time,
                reason=candidate.reason,
            )
            for candidate in candidates
        ]

    monkeypatch.setattr(nodes, "render_candidates", fake_render)
    result = build_agent_graph().invoke(
        {
            "video_path": str(workspace.source_video_path),
            "domain": "lecture",
            "highlight_count": 3,
            "transcript_source": "whisper",
            "burn_subtitles": False,
        }
    )

    assert result["features"]["mode"] == "naive_baseline"
    assert len(result["highlights"]) == 3
    assert len(result["rendered_highlights"]) == 3
    assert len(result["reasoning"]) == 3
    assert calls == {"transcript_source": "whisper", "burn_subtitles": False}
