from pathlib import Path

import pytest
from pydantic import ValidationError

from highlight_agent.schemas import (
    HighlightCandidate,
    MediaWorkspace,
    TranscriptDocument,
    TranscriptSegment,
    TranscriptWord,
)


def test_media_workspace_rejects_path_separator_in_video_id() -> None:
    with pytest.raises(ValidationError, match="path separators"):
        MediaWorkspace(
            video_id="bad/id",
            source_type="local",
            original_input="video.mp4",
            source_video_path=Path("video.mp4"),
            audio_path=Path("output/audio.wav"),
            transcript_path=Path("output/transcript.json"),
        )


def test_transcript_document_accepts_sorted_word_timestamps() -> None:
    document = TranscriptDocument(
        video_id="abcdefghijk",
        language="en",
        source="whisper",
        duration=60,
        segments=[
            TranscriptSegment(
                id=0,
                start=1,
                end=3,
                text="Hello world",
                words=[
                    TranscriptWord(start=1, end=1.5, text="Hello"),
                    TranscriptWord(start=1.5, end=2, text="world"),
                ],
            )
        ],
    )
    assert document.segments[0].words[1].text == "world"


def test_transcript_document_accepts_explicit_no_speech_benchmark() -> None:
    document = TranscriptDocument(
        video_id="benchmark-video",
        language="und",
        source="whisper",
        duration=60,
        segments=[],
    )

    assert document.segments == []


@pytest.mark.parametrize("start,end", [(0, 29.9), (0, 90.1), (50, 49)])
def test_highlight_candidate_enforces_mvp_duration(start: float, end: float) -> None:
    with pytest.raises(ValidationError):
        HighlightCandidate(
            candidate_id="candidate-1",
            start_time=start,
            end_time=end,
            score=1,
            reason="Test candidate",
        )
