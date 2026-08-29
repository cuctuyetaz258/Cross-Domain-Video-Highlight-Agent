"""Canh biên highlight theo transcript và khoảng lặng audio"""

from dataclasses import dataclass
from typing import Literal

from highlight_agent.schemas import (
    BoundaryAdjustment,
    HighlightCandidate,
    TimeInterval,
    TranscriptDocument,
)

BoundarySource = Literal["punctuation", "silence", "segment_fallback", "original"]


@dataclass(frozen=True)
class BoundaryConfig:
    """Cấu hình cho việc tìm mốc cắt tự nhiên"""

    search_radius_seconds: float = 3.0
    end_search_radius_seconds: float = 8.0
    min_duration_seconds: float = 30.0
    max_duration_seconds: float = 90.0


DEFAULT_BOUNDARY_CONFIG = BoundaryConfig()
BOUNDARY_EPSILON_SECONDS = 1e-6


def _sentence_boundaries(transcript: TranscriptDocument) -> tuple[list[float], list[float]]:
    words = [word for segment in transcript.segments for word in segment.words]
    starts: set[float] = set()
    ends: set[float] = set()
    if words:
        starts.add(words[0].start)
    for index, word in enumerate(words):
        if word.text.rstrip().endswith((".", "?", "!")):
            ends.add(word.end)
            if index + 1 < len(words):
                starts.add(words[index + 1].start)

    # Whisper/caption segments can retain punctuation in ``segment.text`` even
    # when the corresponding word list is incomplete or drops that punctuation.
    # Keep those segment-level boundaries so an end cut never falls back to a
    # timestamp inside the final spoken sentence.
    punctuated_segments = [
        segment
        for segment in transcript.segments
        if segment.text.rstrip().endswith((".", "?", "!"))
    ]
    if punctuated_segments and transcript.segments:
        first_segment = transcript.segments[0]
        starts.add(
            first_segment.words[0].start
            if first_segment.words
            else first_segment.start
        )
    for index, segment in enumerate(transcript.segments):
        if segment.text.rstrip().endswith((".", "?", "!")):
            ends.add(segment.end)
            if index + 1 < len(transcript.segments):
                next_segment = transcript.segments[index + 1]
                starts.add(
                    next_segment.words[0].start
                    if next_segment.words
                    else next_segment.start
                )
    return sorted(starts), sorted(ends)


def _segment_boundaries(transcript: TranscriptDocument) -> tuple[list[float], list[float]]:
    return (
        [segment.start for segment in transcript.segments],
        [segment.end for segment in transcript.segments],
    )


def _start_boundary(target: float, boundaries: list[float], radius: float) -> float | None:
    before = [boundary for boundary in boundaries if target - radius <= boundary <= target]
    if before:
        return max(before)
    after = [boundary for boundary in boundaries if target < boundary <= target + radius]
    return min(after) if after else None


def _end_boundary(target: float, boundaries: list[float], radius: float) -> float | None:
    """Return only a future end boundary so speech is never truncated."""

    after = [boundary for boundary in boundaries if target <= boundary <= target + radius]
    return min(after) if after else None


def _nearest_silence_start(boundary: float, silence_intervals: list[TimeInterval], radius: float) -> float | None:
    candidates = [interval.end for interval in silence_intervals if boundary - radius <= interval.end <= boundary]
    return max(candidates) if candidates else None


def _nearest_silence_end(boundary: float, silence_intervals: list[TimeInterval], radius: float) -> float | None:
    candidates = [interval.start for interval in silence_intervals if boundary <= interval.start <= boundary + radius]
    return min(candidates) if candidates else None


def _proposed_start(
    candidate: HighlightCandidate,
    transcript: TranscriptDocument,
    silence_intervals: list[TimeInterval],
    config: BoundaryConfig,
) -> tuple[float, BoundarySource, str]:
    sentence_starts, _ = _sentence_boundaries(transcript)
    boundary = _start_boundary(candidate.start_time, sentence_starts, config.search_radius_seconds)
    source: BoundarySource = "punctuation"
    if boundary is None:
        segment_starts, _ = _segment_boundaries(transcript)
        boundary = _start_boundary(candidate.start_time, segment_starts, config.search_radius_seconds)
        source = "segment_fallback"
    if boundary is None:
        return candidate.start_time, "original", "Không tìm được mốc mở đầu phù hợp trong bán kính tìm kiếm"

    silence_start = _nearest_silence_start(boundary, silence_intervals, config.search_radius_seconds)
    if silence_start is not None:
        return silence_start, "silence", "Bắt đầu sau khoảng lặng liền trước ranh giới câu"
    if source == "punctuation":
        return boundary, source, "Bắt đầu tại ranh giới câu gần nhất"
    return boundary, source, "Bắt đầu tại biên transcript segment gần nhất"


def _proposed_end(
    candidate: HighlightCandidate,
    transcript: TranscriptDocument,
    silence_intervals: list[TimeInterval],
    config: BoundaryConfig,
) -> tuple[float, BoundarySource, str]:
    _, sentence_ends = _sentence_boundaries(transcript)
    boundary = _end_boundary(
        candidate.end_time,
        sentence_ends,
        config.end_search_radius_seconds,
    )
    source: BoundarySource = "punctuation"
    if boundary is None:
        _, segment_ends = _segment_boundaries(transcript)
        boundary = _end_boundary(
            candidate.end_time,
            segment_ends,
            config.end_search_radius_seconds,
        )
        source = "segment_fallback"
    if boundary is None:
        silence_end = _nearest_silence_end(
            candidate.end_time,
            silence_intervals,
            config.end_search_radius_seconds,
        )
        if silence_end is not None:
            return silence_end, "silence", "Kết thúc tại khoảng lặng đầu tiên sau mốc đề xuất"
        return (
            candidate.end_time,
            "original",
            "Không tìm được mốc kết thúc phía sau trong bán kính tìm kiếm",
        )

    silence_end = _nearest_silence_end(boundary, silence_intervals, config.search_radius_seconds)
    if silence_end is not None:
        return silence_end, "silence", "Kết thúc trước khoảng lặng liền sau ranh giới câu"
    if source == "punctuation":
        return boundary, source, "Kết thúc tại ranh giới câu gần nhất"
    return boundary, source, "Kết thúc tại biên transcript segment gần nhất"


def _is_valid_range(start: float, end: float, video_duration: float, config: BoundaryConfig) -> bool:
    duration = end - start
    return (
        0 <= start < end <= video_duration + BOUNDARY_EPSILON_SECONDS
        and config.min_duration_seconds - BOUNDARY_EPSILON_SECONDS
        <= duration
        <= config.max_duration_seconds + BOUNDARY_EPSILON_SECONDS
    )


def refine_candidate_boundary(
    candidate: HighlightCandidate,
    transcript: TranscriptDocument,
    silence_intervals: list[TimeInterval],
    *,
    video_duration: float,
    config: BoundaryConfig = DEFAULT_BOUNDARY_CONFIG,
) -> tuple[HighlightCandidate, BoundaryAdjustment]:
    """Canh biên một candidate mà vẫn giữ giới hạn thời lượng highlight"""

    if video_duration <= 0:
        raise ValueError("video_duration must be positive")
    if config.search_radius_seconds < 0:
        raise ValueError("search_radius_seconds must be non-negative")
    if config.end_search_radius_seconds < 0:
        raise ValueError("end_search_radius_seconds must be non-negative")

    proposed_start, start_source, start_reason = _proposed_start(
        candidate,
        transcript,
        silence_intervals,
        config,
    )
    proposed_end, end_source, end_reason = _proposed_end(
        candidate,
        transcript,
        silence_intervals,
        config,
    )

    options = [
        (proposed_start, proposed_end, True, True),
        (proposed_start, candidate.end_time, True, False),
        (candidate.start_time, proposed_end, False, True),
        (candidate.start_time, candidate.end_time, False, False),
    ]
    valid_options = [option for option in options if _is_valid_range(option[0], option[1], video_duration, config)]
    if not valid_options:
        raise ValueError("candidate boundaries must stay inside the video and satisfy highlight duration")
    start, end, used_start, used_end = min(
        valid_options,
        key=lambda option: (
            -(int(option[2]) + int(option[3])),
            abs(option[0] - candidate.start_time) + abs(option[1] - candidate.end_time),
        ),
    )

    if not used_start:
        start_source = "original"
        start_reason = "Giữ mốc đề xuất để clip không vi phạm thời lượng hoặc phạm vi video"
    if not used_end:
        end_source = "original"
        end_reason = "Giữ mốc đề xuất để clip không vi phạm thời lượng hoặc phạm vi video"

    refined = candidate.model_copy(update={"start_time": round(start, 3), "end_time": round(end, 3)})
    adjustment = BoundaryAdjustment(
        candidate_id=candidate.candidate_id,
        original_start_time=candidate.start_time,
        original_end_time=candidate.end_time,
        start_time=refined.start_time,
        end_time=refined.end_time,
        start_source=start_source,
        end_source=end_source,
        start_reason=start_reason,
        end_reason=end_reason,
    )
    return refined, adjustment


def refine_candidate_boundaries(
    candidates: list[HighlightCandidate],
    transcript: TranscriptDocument,
    silence_intervals: list[TimeInterval],
    *,
    video_duration: float,
    config: BoundaryConfig = DEFAULT_BOUNDARY_CONFIG,
) -> tuple[list[HighlightCandidate], list[BoundaryAdjustment]]:
    """Canh biên độc lập từng candidate mà không khử overlap giữa các candidate"""

    results = [
        refine_candidate_boundary(
            candidate,
            transcript,
            silence_intervals,
            video_duration=video_duration,
            config=config,
        )
        for candidate in candidates
    ]
    return [candidate for candidate, _ in results], [adjustment for _, adjustment in results]
