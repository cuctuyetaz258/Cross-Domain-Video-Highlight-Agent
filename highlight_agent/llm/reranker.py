"""Hybrid LTR + LLM reranking, cache và boundary validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from highlight_agent.schemas import (
    HighlightCandidate,
    LLMHighlightAssessment,
    LLMHighlightAssessmentBatch,
    LLMRunInfo,
    TranscriptDocument,
)

from .client import PROMPT_VERSION, AssessmentClient
from .context import build_candidate_contexts, has_usable_transcript
from .fusion import FUSION_METHOD, fuse_ranked_scores, percentile_rank


def _normalize_scores(candidates: list[HighlightCandidate]) -> dict[str, float]:
    values = fuse_ranked_scores(
        [candidate.score for candidate in candidates],
        [0.0 for _ in candidates],
        alpha=1.0,
    )
    return {
        candidate.candidate_id: float(value)
        for candidate, value in zip(candidates, values)
    }


def _validate_assessment_coverage(
    batch: LLMHighlightAssessmentBatch,
    candidates: list[HighlightCandidate],
) -> None:
    expected = {candidate.candidate_id for candidate in candidates}
    actual = {assessment.candidate_id for assessment in batch.assessments}
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ValueError(f"LLM assessment IDs mismatch; missing={missing}, unknown={unknown}")


def hybrid_rerank(
    candidates: list[HighlightCandidate],
    assessments: list[LLMHighlightAssessment],
    *,
    ltr_weight: float = 0.60,
) -> list[HighlightCandidate]:
    """Chuẩn hóa LTR rồi kết hợp semantic quality về thang 0–10."""

    if not candidates:
        return []
    if not 0 <= ltr_weight <= 1:
        raise ValueError("ltr_weight must be between 0 and 1")
    assessment_map = {item.candidate_id: item for item in assessments}
    if set(assessment_map) != {candidate.candidate_id for candidate in candidates}:
        raise ValueError("every candidate must have exactly one LLM assessment")

    ltr_scores = _normalize_scores(candidates)
    llm_scores = {
        candidate.candidate_id: float(assessment_map[candidate.candidate_id].semantic_quality())
        for candidate in candidates
    }
    llm_ranks = {
        candidate.candidate_id: float(rank)
        for candidate, rank in zip(
            candidates,
            percentile_rank([llm_scores[candidate.candidate_id] for candidate in candidates]),
        )
    }
    hybrid_scores = fuse_ranked_scores(
        [candidate.score for candidate in candidates],
        [llm_scores[candidate.candidate_id] for candidate in candidates],
        alpha=ltr_weight,
    )
    reranked: list[HighlightCandidate] = []
    for candidate, hybrid_score in zip(candidates, hybrid_scores):
        assessment = assessment_map[candidate.candidate_id]
        semantic_quality = assessment.semantic_quality()
        reranked.append(
            candidate.model_copy(
                update={
                    "score": round(hybrid_score * 10.0, 6),
                    "reason": (
                        f"{candidate.reason} LLM semantic assessment: {assessment.evidence}"
                    ),
                    "signals": {
                        **candidate.signals,
                        "ltr_score_original": candidate.score,
                        "ltr_score_normalized": round(ltr_scores[candidate.candidate_id], 6),
                        "llm_semantic_quality": round(semantic_quality, 6),
                        "llm_score_rank": round(llm_ranks[candidate.candidate_id], 6),
                        "fusion_method": FUSION_METHOD,
                        "fusion_alpha": ltr_weight,
                    },
                }
            )
        )
    return sorted(reranked, key=lambda candidate: candidate.score, reverse=True)


def _known_boundaries(transcript: TranscriptDocument) -> tuple[list[float], list[float]]:
    starts = [segment.start for segment in transcript.segments]
    ends = [segment.end for segment in transcript.segments]
    words = [word for segment in transcript.segments for word in segment.words]
    if words:
        starts.append(words[0].start)
        for index, word in enumerate(words):
            if word.text.rstrip().endswith((".", "?", "!")):
                ends.append(word.end)
                if index + 1 < len(words):
                    starts.append(words[index + 1].start)
    return starts, ends


def _snap(value: float, boundaries: list[float], tolerance: float) -> float | None:
    if not boundaries:
        return None
    nearest = min(boundaries, key=lambda boundary: abs(boundary - value))
    return nearest if abs(nearest - value) <= tolerance else None


def apply_validated_boundaries(
    candidates: list[HighlightCandidate],
    assessments: list[LLMHighlightAssessment],
    transcript: TranscriptDocument,
    *,
    timestamp_tolerance: float = 0.35,
    max_adjustment_seconds: float = 15.0,
) -> tuple[list[HighlightCandidate], list[str]]:
    """Chỉ nhận đề xuất gần timestamp thật và vẫn thỏa contract 30–90 giây."""

    assessment_map = {item.candidate_id: item for item in assessments}
    starts, ends = _known_boundaries(transcript)
    accepted: list[str] = []
    result: list[HighlightCandidate] = []
    for candidate in candidates:
        assessment = assessment_map.get(candidate.candidate_id)
        if (
            assessment is None
            or assessment.suggested_start_time is None
            or assessment.suggested_end_time is None
        ):
            result.append(candidate)
            continue
        start = _snap(assessment.suggested_start_time, starts, timestamp_tolerance)
        end = _snap(assessment.suggested_end_time, ends, timestamp_tolerance)
        if start is None or end is None:
            result.append(candidate)
            continue
        duration = end - start
        valid = (
            0 <= start < end <= transcript.duration + 1e-6
            and 30 <= duration <= 90
            and abs(start - candidate.start_time) <= max_adjustment_seconds
            and abs(end - candidate.end_time) <= max_adjustment_seconds
        )
        if not valid:
            result.append(candidate)
            continue
        result.append(
            candidate.model_copy(
                update={
                    "start_time": round(start, 3),
                    "end_time": round(end, 3),
                }
            )
        )
        accepted.append(candidate.candidate_id)
    original_map = {candidate.candidate_id: candidate for candidate in candidates}
    result_map = {candidate.candidate_id: candidate for candidate in result}
    for candidate_id in list(accepted):
        proposed = result_map[candidate_id]
        original = original_map[candidate_id]
        for other_id, other in result_map.items():
            if other_id == candidate_id:
                continue
            other_original = original_map[other_id]
            proposed_overlap = max(
                0.0,
                min(proposed.end_time, other.end_time)
                - max(proposed.start_time, other.start_time),
            )
            original_overlap = max(
                0.0,
                min(original.end_time, other_original.end_time)
                - max(original.start_time, other_original.start_time),
            )
            if proposed_overlap > original_overlap + 1e-6:
                result_map[candidate_id] = original
                accepted.remove(candidate_id)
                break
    return [result_map[candidate.candidate_id] for candidate in result], accepted


def _cache_key(
    *,
    contexts: list[dict],
    provider: str,
    model: str,
    domain: str,
    checkpoint_fingerprint: str,
) -> str:
    payload = {
        "prompt_version": PROMPT_VERSION,
        "provider": provider,
        "model": model,
        "domain": domain,
        "checkpoint_fingerprint": checkpoint_fingerprint,
        "contexts": contexts,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


def rerank_candidates(
    candidates: list[HighlightCandidate],
    transcript: TranscriptDocument,
    *,
    domain: str,
    client: AssessmentClient,
    cache_dir: str | Path,
    ltr_weight: float = 0.60,
    checkpoint_fingerprint: str = "unknown",
) -> tuple[list[HighlightCandidate], list[LLMHighlightAssessment], LLMRunInfo]:
    """Chạy/cache assessment và trả candidate đã hybrid-rerank."""

    if not candidates:
        raise ValueError("LLM reranker requires at least one candidate")
    contexts = build_candidate_contexts(transcript, candidates)
    usable_contexts = [context for context in contexts if has_usable_transcript(context)]
    usable_ids = {context.candidate_id for context in usable_contexts}
    usable_candidates = [
        candidate for candidate in candidates if candidate.candidate_id in usable_ids
    ]
    unavailable_candidates = [
        candidate for candidate in candidates if candidate.candidate_id not in usable_ids
    ]
    if not usable_candidates:
        raise ValueError(
            "no candidate has usable transcript content; OpenAI reranking was skipped"
        )
    dumped_contexts = [context.model_dump(mode="json") for context in usable_contexts]
    key = _cache_key(
        contexts=dumped_contexts,
        provider=client.provider,
        model=client.model,
        domain=domain,
        checkpoint_fingerprint=checkpoint_fingerprint,
    )
    cache_path = Path(cache_dir) / f"assessments_{key}.json"
    cache_hit = cache_path.is_file()
    batch: LLMHighlightAssessmentBatch | None = None
    if cache_hit:
        try:
            batch = LLMHighlightAssessmentBatch.model_validate_json(
                cache_path.read_text(encoding="utf-8")
            )
            _validate_assessment_coverage(batch, usable_candidates)
        except Exception:  # noqa: BLE001
            cache_hit = False
            batch = None
    if batch is None:
        batch = client.assess(usable_contexts, domain=domain)
        _validate_assessment_coverage(batch, usable_candidates)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = cache_path.with_suffix(".json.tmp")
        temporary_path.write_text(batch.model_dump_json(indent=2), encoding="utf-8")
        temporary_path.replace(cache_path)
    reranked = hybrid_rerank(
        usable_candidates,
        batch.assessments,
        ltr_weight=ltr_weight,
    )
    reranked.extend(
        sorted(unavailable_candidates, key=lambda candidate: candidate.score, reverse=True)
    )
    info = LLMRunInfo(
        enabled=True,
        applied=True,
        provider=client.provider,
        model=client.model,
        prompt_version=PROMPT_VERSION,
        cache_hit=cache_hit,
        assessed_count=len(batch.assessments),
    )
    return reranked, batch.assessments, info
