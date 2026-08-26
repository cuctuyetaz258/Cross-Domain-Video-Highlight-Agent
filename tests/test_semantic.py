import numpy as np

from highlight_agent.features.semantic import extract_windowed_semantic_features
from highlight_agent.schemas import TranscriptDocument, TranscriptSegment


class FakeEncoder:
    def encode(self, sentences: list[str], **kwargs) -> np.ndarray:
        vectors = []
        for sentence in sentences:
            if "important" in sentence.casefold() or "quan trọng" in sentence.casefold():
                vectors.append([1.0, 0.0])
            else:
                vectors.append([0.0, 1.0])
        return np.asarray(vectors, dtype=np.float32)


def _transcript() -> TranscriptDocument:
    return TranscriptDocument(
        video_id="abcdefghijk",
        language="en",
        source="whisper",
        duration=65.0,
        segments=[
            TranscriptSegment(id=0, start=0, end=10, text="The key point is an important conclusion"),
            TranscriptSegment(id=1, start=32, end=42, text="Một ví dụ đơn giản và điều quan trọng cần nhớ"),
        ],
    )


def test_semantic_features_follow_window_schedule_and_detect_cues() -> None:
    results = extract_windowed_semantic_features(
        _transcript(),
        window_seconds=30.0,
        hop_seconds=30.0,
        encoder=FakeEncoder(),
    )

    assert [(item.start, item.end) for item in results] == [(0.0, 30.0), (30.0, 60.0), (60.0, 65.0)]
    assert "the key point" in results[0].features.cue_phrases
    assert "điều quan trọng" in results[1].features.cue_phrases
    assert results[2].features.text_coverage == 0.0
    assert results[2].features.raw_score == 0.0


def test_semantic_features_can_follow_audio_duration_beyond_transcript() -> None:
    results = extract_windowed_semantic_features(
        _transcript(),
        window_seconds=30.0,
        hop_seconds=30.0,
        duration=90.0,
        encoder=FakeEncoder(),
    )

    assert len(results) == 3
    assert results[-1].end == 90.0
    for result in results:
        values = result.features.model_dump(exclude={"cue_phrases"})
        assert all(np.isfinite(value) for value in values.values())
        assert 0.0 <= result.features.raw_score <= 1.0
