from pathlib import Path
from types import SimpleNamespace

from highlight_agent import backend
from highlight_agent.boundary import refine_candidate_boundary
from highlight_agent.schemas import (
    HighlightCandidate,
    MediaWorkspace,
    TimeInterval,
    TranscriptDocument,
    TranscriptSegment,
    TranscriptWord,
)


def _candidate(*, start: float = 33.0, end: float = 65.0) -> HighlightCandidate:
    return HighlightCandidate(
        candidate_id="candidate-01",
        start_time=start,
        end_time=end,
        score=8.0,
        reason="Test boundary",
    )


def _word_transcript() -> TranscriptDocument:
    return TranscriptDocument(
        video_id="abcdefghijk",
        language="en",
        source="whisper",
        duration=100.0,
        segments=[
            TranscriptSegment(
                id=0,
                start=31.0,
                end=32.0,
                text="First sentence.",
                words=[
                    TranscriptWord(start=31.0, end=31.4, text="First"),
                    TranscriptWord(start=31.4, end=32.0, text="sentence."),
                ],
            ),
            TranscriptSegment(
                id=1,
                start=65.5,
                end=66.0,
                text="Final sentence.",
                words=[
                    TranscriptWord(start=65.5, end=65.7, text="Final"),
                    TranscriptWord(start=65.7, end=66.0, text="sentence."),
                ],
            ),
        ],
    )


def test_boundary_uses_sentence_boundaries_and_adjacent_silence() -> None:
    refined, adjustment = refine_candidate_boundary(
        _candidate(),
        _word_transcript(),
        [TimeInterval(start=30.6, end=31.0), TimeInterval(start=66.0, end=66.3)],
        video_duration=100.0,
    )

    assert (refined.start_time, refined.end_time) == (31.0, 66.0)
    assert (adjustment.start_source, adjustment.end_source) == ("silence", "silence")


def test_boundary_falls_back_to_nearest_segment_when_words_are_missing() -> None:
    transcript = TranscriptDocument(
        video_id="abcdefghijk",
        language="en",
        source="youtube_caption",
        duration=100.0,
        segments=[
            TranscriptSegment(id=0, start=31.0, end=32.0, text="First caption"),
            TranscriptSegment(id=1, start=65.5, end=66.0, text="Final caption"),
        ],
    )

    refined, adjustment = refine_candidate_boundary(
        _candidate(),
        transcript,
        [],
        video_duration=100.0,
    )

    assert (refined.start_time, refined.end_time) == (31.0, 66.0)
    assert (adjustment.start_source, adjustment.end_source) == ("segment_fallback", "segment_fallback")


def test_boundary_extends_to_next_segment_instead_of_using_previous_sentence_end() -> None:
    transcript = TranscriptDocument(
        video_id="abcdefghijk",
        language="en",
        source="whisper",
        duration=200.0,
        segments=[
            TranscriptSegment(
                id=0,
                start=126.0,
                end=129.0,
                text="The previous sentence.",
                words=[
                    TranscriptWord(start=126.0, end=128.0, text="previous"),
                    TranscriptWord(start=128.0, end=129.0, text="sentence."),
                ],
            ),
            TranscriptSegment(
                id=1,
                start=129.2,
                end=134.5,
                text="This sentence must be completed.",
                words=[
                    TranscriptWord(start=129.2, end=131.0, text="This"),
                    TranscriptWord(start=131.0, end=134.5, text="completed"),
                ],
            ),
        ],
    )

    refined, adjustment = refine_candidate_boundary(
        _candidate(start=100.0, end=130.0),
        transcript,
        [],
        video_duration=200.0,
    )

    assert refined.end_time == 134.5
    assert adjustment.end_source == "punctuation"


def test_boundary_never_moves_end_backward_to_previous_punctuation() -> None:
    transcript = TranscriptDocument(
        video_id="abcdefghijk",
        language="en",
        source="whisper",
        duration=200.0,
        segments=[
            TranscriptSegment(
                id=0,
                start=120.0,
                end=129.0,
                text="The only complete sentence.",
                words=[TranscriptWord(start=128.0, end=129.0, text="sentence.")],
            )
        ],
    )

    refined, adjustment = refine_candidate_boundary(
        _candidate(start=100.0, end=130.0),
        transcript,
        [],
        video_duration=200.0,
    )

    assert refined.end_time == 130.0
    assert adjustment.end_source == "original"


def test_boundary_keeps_original_start_when_shift_would_make_clip_too_short() -> None:
    transcript = TranscriptDocument(
        video_id="abcdefghijk",
        language="en",
        source="whisper",
        duration=200.0,
        segments=[
            TranscriptSegment(
                id=0,
                start=103.0,
                end=104.0,
                text="Complete sentence.",
                words=[
                    TranscriptWord(start=103.0, end=103.5, text="Complete"),
                    TranscriptWord(start=103.5, end=104.0, text="sentence."),
                ],
            )
        ],
    )

    refined, adjustment = refine_candidate_boundary(
        _candidate(start=100.0, end=130.0),
        transcript,
        [],
        video_duration=200.0,
    )

    assert refined.start_time == 100.0
    assert adjustment.start_source == "original"
    assert "thời lượng" in adjustment.start_reason


def test_boundary_keeps_original_when_no_transcript_boundary_is_nearby() -> None:
    refined, adjustment = refine_candidate_boundary(
        _candidate(start=35.0, end=70.0),
        _word_transcript(),
        [],
        video_duration=100.0,
    )

    assert (refined.start_time, refined.end_time) == (35.0, 70.0)
    assert adjustment.start_source == "original"
    assert adjustment.end_source == "original"


def test_boundary_accepts_minimum_duration_with_float_roundoff() -> None:
    candidate = _candidate(start=255.4, end=285.4)

    refined, adjustment = refine_candidate_boundary(
        candidate,
        _word_transcript().model_copy(update={"duration": 320.555193}),
        [],
        video_duration=320.555193,
    )

    assert (refined.start_time, refined.end_time) == (255.4, 285.4)
    assert refined.end_time - refined.start_time >= 30.0 - 1e-6
    assert adjustment.start_source == "original"
    assert adjustment.end_source == "original"


def test_backend_extracts_silence_when_feature_timeline_does_not_exist(tmp_path: Path, monkeypatch) -> None:
    transcript = _word_transcript()
    workspace_dir = tmp_path / "output" / transcript.video_id
    workspace_dir.mkdir(parents=True)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"fixture")
    workspace = MediaWorkspace(
        video_id=transcript.video_id,
        source_type="local",
        original_input=str(source),
        source_video_path=source,
        audio_path=workspace_dir / "audio.wav",
        transcript_path=workspace_dir / "transcript.json",
    )
    workspace.transcript_path.write_text(transcript.model_dump_json(), encoding="utf-8")
    monkeypatch.setattr(
        backend,
        "extract_acoustic_features",
        lambda path: SimpleNamespace(silence_intervals=[TimeInterval(start=30.6, end=31.0)]),
    )

    refined, adjustments = backend.refine_candidates_for_render(workspace, [_candidate()])

    assert refined[0].start_time == 31.0
    assert adjustments[0].start_source == "silence"
