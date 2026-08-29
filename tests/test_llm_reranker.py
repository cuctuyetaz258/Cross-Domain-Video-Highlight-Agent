from pathlib import Path

import pytest

from highlight_agent.llm.context import build_candidate_contexts
from highlight_agent.llm.reranker import (
    apply_validated_boundaries,
    hybrid_rerank,
    rerank_candidates,
)
from highlight_agent.schemas import (
    HighlightCandidate,
    LLMHighlightAssessment,
    LLMHighlightAssessmentBatch,
    TranscriptDocument,
    TranscriptSegment,
)


def _transcript() -> TranscriptDocument:
    return TranscriptDocument(
        video_id="video-01",
        language="vi",
        source="whisper",
        duration=120,
        segments=[
            TranscriptSegment(id=0, start=0, end=29, text="Bối cảnh mở đầu."),
            TranscriptSegment(id=1, start=29, end=61, text="Đây là ý chính hoàn chỉnh."),
            TranscriptSegment(id=2, start=61, end=90, text="Phần giải thích tiếp theo."),
            TranscriptSegment(id=3, start=90, end=120, text="Kết luận."),
        ],
    )


def _candidate(candidate_id: str, start: float, end: float, score: float) -> HighlightCandidate:
    return HighlightCandidate(
        candidate_id=candidate_id,
        start_time=start,
        end_time=end,
        score=score,
        reason="LTR candidate",
        signals={"audio": 0.5},
    )


def _assessment(
    candidate_id: str,
    *,
    semantic: float,
    suggested_start: float | None = None,
    suggested_end: float | None = None,
) -> LLMHighlightAssessment:
    return LLMHighlightAssessment(
        candidate_id=candidate_id,
        semantic_relevance=semantic,
        standalone_value=semantic,
        completeness=semantic,
        hook_strength=semantic,
        shareability=semantic,
        title=f"Title {candidate_id}",
        summary=f"Summary {candidate_id}",
        evidence="Transcript evidence",
        suggested_start_time=suggested_start,
        suggested_end_time=suggested_end,
        risk_flags=[],
    )


def test_context_builder_uses_local_before_core_after() -> None:
    contexts = build_candidate_contexts(
        _transcript(),
        [_candidate("c1", 30, 60, 7)],
        before_seconds=10,
        after_seconds=10,
    )

    assert len(contexts) == 1
    assert "Bối cảnh mở đầu" in contexts[0].before
    assert "ý chính hoàn chỉnh" in contexts[0].core
    assert "giải thích tiếp theo" in contexts[0].after
    assert "Kết luận" not in contexts[0].after


def test_hybrid_rerank_combines_normalized_ltr_and_semantic_scores() -> None:
    candidates = [
        _candidate("physical", 0, 30, 10),
        _candidate("semantic", 60, 90, 1),
    ]
    reranked = hybrid_rerank(
        candidates,
        [_assessment("physical", semantic=0), _assessment("semantic", semantic=1)],
        ltr_weight=0.40,
    )

    assert reranked[0].candidate_id == "semantic"
    assert reranked[0].signals["ltr_score_original"] == 1
    assert reranked[0].signals["llm_semantic_quality"] == pytest.approx(1)


def test_boundary_suggestion_must_match_real_transcript_timestamps() -> None:
    candidate = _candidate("c1", 30, 60, 7)
    accepted, accepted_ids = apply_validated_boundaries(
        [candidate],
        [_assessment("c1", semantic=1, suggested_start=29, suggested_end=61)],
        _transcript(),
    )
    rejected, rejected_ids = apply_validated_boundaries(
        [candidate],
        [_assessment("c1", semantic=1, suggested_start=28.4, suggested_end=61.6)],
        _transcript(),
    )

    assert (accepted[0].start_time, accepted[0].end_time) == (29, 61)
    assert accepted_ids == ["c1"]
    assert (rejected[0].start_time, rejected[0].end_time) == (30, 60)
    assert rejected_ids == []


def test_boundary_suggestion_is_rejected_when_it_increases_overlap() -> None:
    transcript = TranscriptDocument(
        video_id="video-overlap",
        language="en",
        source="whisper",
        duration=90,
        segments=[
            TranscriptSegment(id=0, start=0, end=31, text="First complete idea."),
            TranscriptSegment(id=1, start=31, end=61, text="Second complete idea."),
            TranscriptSegment(id=2, start=61, end=90, text="Ending."),
        ],
    )
    candidates = [_candidate("c1", 0, 30, 8), _candidate("c2", 30, 60, 7)]
    adjusted, accepted_ids = apply_validated_boundaries(
        candidates,
        [
            _assessment("c1", semantic=1, suggested_start=0, suggested_end=31),
            _assessment("c2", semantic=1),
        ],
        transcript,
    )

    assert (adjusted[0].start_time, adjusted[0].end_time) == (0, 30)
    assert accepted_ids == []


class _FakeClient:
    provider = "fake"
    model = "fake-model"

    def __init__(self) -> None:
        self.calls = 0

    def assess(self, contexts, *, domain):
        self.calls += 1
        return LLMHighlightAssessmentBatch(
            assessments=[_assessment(context.candidate_id, semantic=0.8) for context in contexts]
        )


def test_reranker_skips_api_when_all_candidates_lack_transcript(tmp_path: Path) -> None:
    transcript = TranscriptDocument(
        video_id="video-empty",
        language="en",
        source="whisper",
        duration=120,
        segments=[TranscriptSegment(id=0, start=0, end=10, text="Intro only.")],
    )
    client = _FakeClient()

    with pytest.raises(ValueError, match="OpenAI reranking was skipped"):
        rerank_candidates(
            [_candidate("c1", 30, 60, 7), _candidate("c2", 61, 91, 6)],
            transcript,
            domain="lecture",
            client=client,
            cache_dir=tmp_path,
        )

    assert client.calls == 0
    assert list(tmp_path.glob("assessments_*.json")) == []


def test_reranker_only_sends_candidates_with_core_transcript(tmp_path: Path) -> None:
    transcript = TranscriptDocument(
        video_id="video-partial",
        language="en",
        source="whisper",
        duration=120,
        segments=[TranscriptSegment(id=0, start=30, end=60, text="Usable core.")],
    )
    client = _FakeClient()

    ranked, assessments, info = rerank_candidates(
        [_candidate("usable", 30, 60, 7), _candidate("missing", 61, 91, 9)],
        transcript,
        domain="lecture",
        client=client,
        cache_dir=tmp_path,
    )

    assert client.calls == 1
    assert [item.candidate_id for item in assessments] == ["usable"]
    assert [item.candidate_id for item in ranked] == ["usable", "missing"]
    assert info.assessed_count == 1


def test_reranker_caches_structured_assessments_without_raw_context(tmp_path: Path) -> None:
    candidates = [_candidate("c1", 30, 60, 7), _candidate("c2", 61, 91, 6)]
    client = _FakeClient()

    first = rerank_candidates(
        candidates,
        _transcript(),
        domain="lecture",
        client=client,
        cache_dir=tmp_path,
    )
    second = rerank_candidates(
        candidates,
        _transcript(),
        domain="lecture",
        client=client,
        cache_dir=tmp_path,
    )

    assert client.calls == 1
    assert first[2].cache_hit is False
    assert second[2].cache_hit is True
    cache_text = next(tmp_path.glob("assessments_*.json")).read_text(encoding="utf-8")
    assert "Bối cảnh mở đầu" not in cache_text
    assert "Title c1" in cache_text

    cache_path = next(tmp_path.glob("assessments_*.json"))
    cache_path.write_text(
        LLMHighlightAssessmentBatch(
            assessments=[_assessment("c1", semantic=0.8)]
        ).model_dump_json(),
        encoding="utf-8",
    )
    recovered = rerank_candidates(
        candidates,
        _transcript(),
        domain="lecture",
        client=client,
        cache_dir=tmp_path,
    )
    assert client.calls == 2
    assert recovered[2].cache_hit is False


def test_reranker_rejects_missing_assessment(tmp_path: Path) -> None:
    candidates = [_candidate("c1", 30, 60, 7), _candidate("c2", 61, 91, 6)]

    class IncompleteClient(_FakeClient):
        def assess(self, contexts, *, domain):
            return LLMHighlightAssessmentBatch(assessments=[_assessment("c1", semantic=0.8)])

    with pytest.raises(ValueError, match="IDs mismatch"):
        rerank_candidates(
            candidates,
            _transcript(),
            domain="lecture",
            client=IncompleteClient(),
            cache_dir=tmp_path,
        )


def test_reranker_cache_is_bound_to_checkpoint_fingerprint(tmp_path: Path) -> None:
    candidates = [_candidate("c1", 30, 60, 7), _candidate("c2", 61, 91, 6)]
    client = _FakeClient()

    rerank_candidates(
        candidates,
        _transcript(),
        domain="lecture",
        client=client,
        cache_dir=tmp_path,
        checkpoint_fingerprint="checkpoint-a",
    )
    rerank_candidates(
        candidates,
        _transcript(),
        domain="lecture",
        client=client,
        cache_dir=tmp_path,
        checkpoint_fingerprint="checkpoint-b",
    )

    assert client.calls == 2
    assert len(list(tmp_path.glob("assessments_*.json"))) == 2
