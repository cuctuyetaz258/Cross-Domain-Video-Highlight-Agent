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


def test_decide_forwards_aspect_ratio_to_render_candidates(tmp_path: Path, monkeypatch) -> None:
    workspace = _workspace(tmp_path)
    candidate = HighlightCandidate(
        candidate_id="c1",
        start_time=0.0,
        end_time=30.0,
        score=0.9,
        reason="Test",
    )
    rendered = RenderedHighlight(
        candidate_id="c1",
        video_path=workspace.transcript_path.parent / "c1.mp4",
        thumbnail_path=None,
        start_time=0.0,
        end_time=30.0,
        reason="Test",
        aspect_ratio="16:9",
        width=1920,
        height=1080,
    )

    captured = {}

    def fake_refine(ws, candidates):
        return candidates, []

    def fake_render(ws, highlights, *, aspect_ratio="9:16", **kwargs):
        captured["aspect_ratio"] = aspect_ratio
        return [rendered]

    monkeypatch.setattr(nodes, "refine_candidates_for_render", fake_refine)
    monkeypatch.setattr(nodes, "render_candidates", fake_render)

    state = {
        "workspace": workspace,
        "candidates": [candidate, candidate.model_copy(update={"candidate_id": "c2"}), candidate.model_copy(update={"candidate_id": "c3"})],
        "highlight_count": 3,
        "aspect_ratio": "16:9",
        "burn_subtitles": False,
    }
    result = nodes.decide(state)  # type: ignore[arg-type]

    assert captured.get("aspect_ratio") == "16:9"
    assert result["rendered_highlights"][0].aspect_ratio == "16:9"
    assert result["rendered_highlights"][0].width == 1920
    assert result["rendered_highlights"][0].height == 1080


def test_analysis_graph_stops_before_decide(tmp_path: Path, monkeypatch) -> None:
    workspace = _workspace(tmp_path)
    candidate = HighlightCandidate(
        candidate_id="ltr_01",
        start_time=0.0,
        end_time=30.0,
        score=0.9,
        reason="LTR",
    )
    calls: list[str] = []

    monkeypatch.setattr(
        graph_module,
        "preflight",
        lambda state: calls.append("preflight") or {"ltr_checkpoint_info": {"fingerprint": "fixture"}},
    )
    monkeypatch.setattr(
        graph_module,
        "observe",
        lambda state: calls.append("observe") or {"workspace": workspace, "transcript": object()},
    )
    monkeypatch.setattr(
        graph_module,
        "plan",
        lambda state: calls.append("plan") or {"analysis_plan": {"scorer": "ltr_required"}},
    )
    monkeypatch.setattr(
        graph_module,
        "analyze",
        lambda state: calls.append("analyze")
        or {"features": {"mode": "ltr_required"}, "candidates": [candidate]},
    )

    result = graph_module.build_analysis_graph().invoke(
        {
            "video_path": str(workspace.source_video_path),
            "domain": "lecture",
            "ltr_model_path": "checkpoint.pt",
        }
    )

    assert calls == ["preflight", "observe", "plan", "analyze"]
    assert result["candidates"] == [candidate]


def test_explain_generates_user_friendly_offline_reasoning() -> None:
    candidate = HighlightCandidate(
        candidate_id="ltr_01",
        start_time=49.32,
        end_time=84.50,
        score=0.95,
        reason="LTR peak",
    )
    state = {
        "domain": "lecture",
        "highlights": [candidate],
        "features": {"mode": "ltr_required"},
        "llm_assessments": [],
    }
    result = nodes.explain(state)  # type: ignore[arg-type]

    assert len(result["reasoning"]) == 1
    explanation = result["reasoning"][0]["explanation"]
    assert "🎯 Highlight #1" in explanation
    assert "0:49 – 1:24" in explanation
    assert "Semantic explanation unavailable (LLM API key not provided or LLM disabled)" in explanation
    assert "📊 LTR Relative Rank Score**: 95.0%" in explanation
    assert "normalized within this video" in explanation


def test_explain_generates_user_friendly_llm_reasoning() -> None:
    from highlight_agent.schemas import LLMHighlightAssessment

    candidate = HighlightCandidate(
        candidate_id="ltr_01",
        start_time=30.0,
        end_time=65.0,
        score=8.5,
        reason="LTR + LLM",
    )
    assessment = LLMHighlightAssessment(
        candidate_id="ltr_01",
        overall_quality=0.88,
        semantic_relevance=0.9,
        standalone_value=0.85,
        completeness=0.95,
        hook_strength=0.8,
        shareability=0.75,
        title="Introduction to Gradient Descent",
        summary="Clear and structured explanation of gradient descent fundamentals.",
        evidence="The steepest slope points to the local minimum.",
        suggested_start_time=None,
        suggested_end_time=None,
        risk_flags=[],
    )
    state = {
        "domain": "lecture",
        "highlights": [candidate],
        "features": {"mode": "ltr_llm_rerank"},
        "llm_assessments": [assessment],
    }
    result = nodes.explain(state)  # type: ignore[arg-type]

    assert len(result["reasoning"]) == 1
    explanation = result["reasoning"][0]["explanation"]
    assert "💡 Key Insight" in explanation
    assert "Clear and structured explanation" in explanation
    assert '💬 *"The steepest slope points to the local minimum."' in explanation
    assert "0:30 – 1:05" in explanation
    assert "LLM Overall Quality**: 88%" in explanation
