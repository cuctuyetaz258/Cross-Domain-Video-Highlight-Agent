from pathlib import Path

import pytest

pytest.importorskip("langgraph", reason="langgraph không có — skip agent graph tests")

from highlight_agent.agent import graph as graph_module
from highlight_agent.agent import nodes
from highlight_agent.schemas import HighlightCandidate, MediaWorkspace, RenderedHighlight


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


def test_plan_declares_required_ltr_extractors_without_profile_weights() -> None:
    lecture = nodes.plan({"video_path": "video.mp4", "domain": "lecture"})
    podcast = nodes.plan({"video_path": "video.mp4", "domain": "podcast"})

    assert "profile" not in lecture
    assert lecture["analysis_plan"] == {
        "scorer": "ltr_required",
        "scene_extractor": "scenedetect",
        "gesture_extractor": "mediapipe",
        "interaction_extractor": "zero_channel",
    }
    assert podcast["analysis_plan"]["interaction_extractor"] == "pyannote"


def test_plan_rejects_unknown_domain() -> None:
    with pytest.raises(ValueError, match="unsupported domain"):
        nodes.plan({"video_path": "video.mp4", "domain": "unknown"})


def test_full_graph_runs_preflight_before_media_and_keeps_ltr_mode(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workspace = _workspace(tmp_path)
    candidate = HighlightCandidate(
        candidate_id="ltr_01",
        start_time=0.0,
        end_time=30.0,
        score=0.9,
        reason="LTR",
    )
    rendered = RenderedHighlight(
        candidate_id=candidate.candidate_id,
        video_path=workspace.transcript_path.parent / "ltr_01.mp4",
        thumbnail_path=None,
        start_time=candidate.start_time,
        end_time=candidate.end_time,
        reason=candidate.reason,
    )
    calls: list[str] = []

    def fake_preflight(state):
        calls.append("preflight")
        return {"ltr_checkpoint_info": {"fingerprint": "fixture"}}

    def fake_observe(state):
        assert calls == ["preflight"]
        calls.append("observe")
        return {"workspace": workspace, "transcript": object()}

    def fake_plan(state):
        calls.append("plan")
        return {"analysis_plan": {"scorer": "ltr_required"}}

    def fake_analyze(state):
        calls.append("analyze")
        return {"features": {"mode": "ltr_required"}, "candidates": [candidate]}

    def fake_decide(state):
        calls.append("decide")
        return {
            "highlights": [candidate],
            "rendered_highlights": [rendered],
            "features": state["features"],
        }

    def fake_explain(state):
        calls.append("explain")
        return {"reasoning": [{"candidate_id": "ltr_01", "explanation": "LTR"}]}

    monkeypatch.setattr(graph_module, "preflight", fake_preflight)
    monkeypatch.setattr(graph_module, "observe", fake_observe)
    monkeypatch.setattr(graph_module, "plan", fake_plan)
    monkeypatch.setattr(graph_module, "analyze", fake_analyze)
    monkeypatch.setattr(graph_module, "decide", fake_decide)
    monkeypatch.setattr(graph_module, "explain", fake_explain)

    result = graph_module.build_agent_graph().invoke(
        {
            "video_path": str(workspace.source_video_path),
            "domain": "lecture",
            "highlight_count": 3,
            "ltr_model_path": "checkpoint.pt",
        }
    )

    assert calls == ["preflight", "observe", "plan", "analyze", "decide", "explain"]
    assert result["features"]["mode"] == "ltr_required"
    assert result["rendered_highlights"] == [rendered]
