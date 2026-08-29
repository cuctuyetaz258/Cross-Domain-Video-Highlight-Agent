"""Xây transcript context cục bộ cho LLM reranker."""

from __future__ import annotations

from highlight_agent.schemas import (
    CandidateTranscriptContext,
    HighlightCandidate,
    TranscriptDocument,
    TranscriptSegment,
)


def _format_segments(segments: list[TranscriptSegment], *, max_chars: int) -> str:
    lines: list[str] = []
    used = 0
    for segment in segments:
        line = f"[{segment.start:.3f}-{segment.end:.3f}] {segment.text}"
        if lines and used + len(line) + 1 > max_chars:
            break
        if not lines and len(line) > max_chars:
            line = line[:max_chars]
        lines.append(line)
        used += len(line) + 1
    return "\n".join(lines)


def _overlaps(segment: TranscriptSegment, start: float, end: float) -> bool:
    return segment.end > start and segment.start < end


def build_candidate_contexts(
    transcript: TranscriptDocument,
    candidates: list[HighlightCandidate],
    *,
    before_seconds: float = 15.0,
    after_seconds: float = 15.0,
    max_chars_per_section: int = 3000,
) -> list[CandidateTranscriptContext]:
    """Tạo BEFORE/CORE/AFTER theo timestamp thay vì cắt đầu transcript."""

    if before_seconds < 0 or after_seconds < 0:
        raise ValueError("context padding must be non-negative")
    if max_chars_per_section < 100:
        raise ValueError("max_chars_per_section must be at least 100")
    ids = [candidate.candidate_id for candidate in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("candidate IDs must be unique")

    contexts: list[CandidateTranscriptContext] = []
    for candidate in candidates:
        before_start = max(0.0, candidate.start_time - before_seconds)
        after_end = min(transcript.duration, candidate.end_time + after_seconds)
        before = [
            segment
            for segment in transcript.segments
            if _overlaps(segment, before_start, candidate.start_time)
        ]
        core = [
            segment
            for segment in transcript.segments
            if _overlaps(segment, candidate.start_time, candidate.end_time)
        ]
        after = [
            segment
            for segment in transcript.segments
            if _overlaps(segment, candidate.end_time, after_end)
        ]
        core_text = _format_segments(core, max_chars=max_chars_per_section)
        if not core_text:
            core_text = f"[{candidate.start_time:.3f}-{candidate.end_time:.3f}] [NO TRANSCRIPT]"
        contexts.append(
            CandidateTranscriptContext(
                candidate_id=candidate.candidate_id,
                start_time=candidate.start_time,
                end_time=candidate.end_time,
                ltr_score=candidate.score,
                before=_format_segments(before, max_chars=max_chars_per_section),
                core=core_text,
                after=_format_segments(after, max_chars=max_chars_per_section),
            )
        )
    return contexts

