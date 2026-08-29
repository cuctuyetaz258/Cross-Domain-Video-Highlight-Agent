from pathlib import Path

from highlight_agent.agent import nodes
from highlight_agent.schemas import (
    HighlightCandidate,
    LLMHighlightAssessment,
    LLMRunInfo,
    MediaWorkspace,
    RenderedHighlight,
    TranscriptDocument,
    TranscriptSegment,
)


def _candidate(candidate_id: str, start: float, score: float) -> HighlightCandidate:
    return HighlightCandidate(
        candidate_id=candidate_id,
        start_time=start,
        end_time=start + 30,
        score=score,
        reason="LTR candidate",
    )


def _assessment(candidate_id: str, score: float) -> LLMHighlightAssessment:
    return LLMHighlightAssessment(
        candidate_id=candidate_id,
        semantic_relevance=score,
        standalone_value=score,
        completeness=score,
        hook_strength=score,
        shareability=score,
        title=f"Title {candidate_id}",
        summary=f"Summary {candidate_id}",
        evidence="Supported by transcript",
        suggested_start_time=None,
        suggested_end_time=None,
        risk_flags=[],
    )


def _state(tmp_path: Path) -> dict:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    workspace_dir = tmp_path / "output" / "video-01"
    workspace_dir.mkdir(parents=True)
    transcript = TranscriptDocument(
        video_id="video-01",
        language="en",
        source="whisper",
        duration=120,
        segments=[TranscriptSegment(id=0, start=0, end=120, text="A complete transcript segment")],
    )
    return {
        "video_path": str(source),
        "domain": "lecture",
        "workspace": MediaWorkspace(
            video_id="video-01",
            source_type="local",
            original_input=str(source),
            source_video_path=source,
            audio_path=workspace_dir / "audio.wav",
            transcript_path=workspace_dir / "transcript.json",
        ),
        "transcript": transcript,
        "features": {"mode": "ltr_required"},
        "profile": {},
        "highlight_count": 3,
        "candidates": [
            _candidate("c1", 0, 9),
            _candidate("c2", 30, 8),
            _candidate("c3", 60, 7),
        ],
        "llm_provider": "openai",
    }


def test_decide_applies_llm_metadata_to_rendered_highlights(tmp_path: Path, monkeypatch) -> None:
    state = _state(tmp_path)
    assessments = [_assessment("c1", 0.3), _assessment("c2", 1.0), _assessment("c3", 0.5)]

    monkeypatch.setattr(nodes.LLMClientConfig, "from_env", lambda **kwargs: object())
    monkeypatch.setattr(nodes, "OpenAICompatibleAssessmentClient", lambda config: object())
    monkeypatch.setattr(
        nodes,
        "rerank_candidates",
        lambda candidates, transcript, **kwargs: (
            [candidates[1], candidates[2], candidates[0]],
            assessments,
            LLMRunInfo(
                enabled=True,
                applied=True,
                provider="openai",
                model="fake",
                prompt_version="test",
                assessed_count=3,
            ),
        ),
    )
    monkeypatch.setattr(
        nodes,
        "refine_candidates_for_render",
        lambda workspace, candidates: (candidates, []),
    )

    def fake_render(workspace, candidates, *, llm_assessments, **kwargs):
        return [
            RenderedHighlight(
                candidate_id=candidate.candidate_id,
                video_path=workspace.transcript_path.parent / f"{candidate.candidate_id}.mp4",
                start_time=candidate.start_time,
                end_time=candidate.end_time,
                reason=candidate.reason,
                title=llm_assessments[candidate.candidate_id].title,
                summary=llm_assessments[candidate.candidate_id].summary,
            )
            for candidate in candidates
        ]

    monkeypatch.setattr(nodes, "render_candidates", fake_render)
    result = nodes.decide(state)

    assert result["features"]["mode"] == "ltr_llm_rerank"
    assert result["highlights"][0].candidate_id == "c2"
    assert result["rendered_highlights"][0].title == "Title c2"
    assert result["llm_run"].applied is True


def test_decide_falls_back_to_ltr_when_provider_fails(tmp_path: Path, monkeypatch) -> None:
    state = _state(tmp_path)
    monkeypatch.setattr(
        nodes.LLMClientConfig,
        "from_env",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("missing key")),
    )
    monkeypatch.setattr(
        nodes,
        "refine_candidates_for_render",
        lambda workspace, candidates: (candidates, []),
    )
    monkeypatch.setattr(nodes, "render_candidates", lambda workspace, candidates, **kwargs: [])

    result = nodes.decide(state)

    assert [item.candidate_id for item in result["highlights"]] == ["c1", "c2", "c3"]
    assert result["features"]["mode"] == "ltr_required"
    assert result["llm_run"].applied is False
    assert "missing key" in result["llm_run"].fallback_reason
