from pathlib import Path

import pytest

pytest.importorskip("langgraph", reason="langgraph không có — skip agent graph tests")

from highlight_agent.agent import build_agent_graph, nodes
from highlight_agent.features.visual import WindowVisualScore
from highlight_agent.schemas import (
    AcousticFeatures,
    FeatureWindow,
    InteractionFeatures,
    MediaWorkspace,
    RenderedHighlight,
    SpeakerTurn,
    TranscriptDocument,
    TranscriptSegment,
)


def _acoustic_features() -> AcousticFeatures:
    return AcousticFeatures(
        duration=360,
        rms_mean=0.1,
        rms_peak=0.2,
        rms_p95=0.18,
        rms_std=0.02,
        voiced_ratio=0.8,
        silence_duration=20,
        silence_ratio=20 / 360,
    )


def _interaction_features() -> InteractionFeatures:
    return InteractionFeatures(
        duration=360,
        speaker_count=2,
        turn_count=12,
        turn_rate_per_minute=2,
        speech_duration=300,
        speech_ratio=300 / 360,
        turns=[
            SpeakerTurn(start=0, end=150, speaker="SPEAKER_00"),
            SpeakerTurn(start=150, end=300, speaker="SPEAKER_01"),
        ],
    )


def _acoustic_windows() -> tuple[AcousticFeatures, list[FeatureWindow]]:
    acoustic = _acoustic_features()
    windows = [
        FeatureWindow(
            start=float(start),
            end=float(start + 30),
            acoustic=_acoustic_features().model_copy(update={"duration": 30.0}),
        )
        for start in range(0, 360, 30)
    ]
    return acoustic, windows


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

    assert lecture["semantic"] == 0.50
    assert podcast["interaction"] == 0.30
    assert sum(lecture.values()) == 1


def test_analyze_naive_baseline_is_reproducible(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(nodes, "extract_windowed_acoustic_features", lambda path, **kwargs: _acoustic_windows())
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
    assert first["features"]["acoustic"]["rms_mean"] == 0.1
    assert Path(first["feature_path"]).is_file()
    assert first["feature_timeline"]["window_seconds"] == 30
    assert len(first["feature_timeline"]["windows"]) == 12


def test_analyze_with_visual_candidates(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(nodes, "extract_windowed_acoustic_features", lambda path, **kwargs: _acoustic_windows())
    fake_visual_scores = [
        WindowVisualScore(start=float(s), end=float(s + 30), motion_score=float(s % 5), method="pixel_diff")
        for s in range(0, 120, 30)
    ]
    monkeypatch.setattr(nodes, "extract_visual_scores", lambda **kwargs: fake_visual_scores)

    state = {
        "video_path": "video.mp4",
        "domain": "lecture",
        "workspace": _workspace(tmp_path),
        "transcript": _transcript(),
        "profile": nodes.PROFILE_WEIGHTS["lecture"],
        "visual_method": "pixel_diff",
    }

    result = nodes.analyze(state)
    assert result["features"]["mode"] == "visual_pixel_diff"
    assert len(result["candidates"]) == 4
    assert result["candidates"][0].signals["visual_motion_raw"] >= 0.0


def test_analyze_adds_interaction_features_for_podcast(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(nodes, "extract_windowed_acoustic_features", lambda path, **kwargs: _acoustic_windows())
    captured = {}

    def fake_interaction(path, *, num_speakers=None):
        captured["num_speakers"] = num_speakers
        return _interaction_features()

    monkeypatch.setattr(nodes, "extract_interaction_features", fake_interaction)
    monkeypatch.setattr(
        nodes,
        "windowed_interaction_features",
        lambda interaction, **kwargs: [_interaction_features().model_copy(update={"duration": 30.0})] * 12,
    )
    state = {
        "video_path": "podcast.mp4",
        "domain": "podcast",
        "workspace": _workspace(tmp_path),
        "transcript": _transcript(),
        "profile": nodes.PROFILE_WEIGHTS["podcast"],
        "known_speaker_count": 2,
    }

    result = nodes.analyze(state)

    assert result["features"]["interaction"]["turn_count"] == 12
    assert captured["num_speakers"] == 2
    assert result["feature_timeline"]["windows"][0]["interaction"]["duration"] == 30


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
    monkeypatch.setattr(nodes, "extract_windowed_acoustic_features", lambda path, **kwargs: _acoustic_windows())
    monkeypatch.setattr(
        nodes,
        "refine_candidates_for_render",
        lambda workspace_arg, candidates: (candidates, []),
    )

    def fake_render(workspace_arg, candidates, *, burn_subtitles=True, **kwargs):
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
