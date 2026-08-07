import json
from pathlib import Path
from types import SimpleNamespace

from highlight_agent.media.transcript import (
    parse_youtube_json3,
    transcribe_with_whisper,
)


def test_parse_youtube_json3_creates_segments_and_word_timestamps(tmp_path: Path) -> None:
    caption = tmp_path / "caption.json3"
    caption.write_text(
        json.dumps(
            {
                "events": [
                    {
                        "tStartMs": 1000,
                        "dDurationMs": 2000,
                        "segs": [
                            {"utf8": "Hello ", "tOffsetMs": 0},
                            {"utf8": "world", "tOffsetMs": 800},
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    document = parse_youtube_json3(caption, video_id="abcdefghijk", duration=10)

    assert document.source == "youtube_caption"
    assert document.segments[0].text == "Hello world"
    assert [word.text for word in document.segments[0].words] == ["Hello", "world"]


def test_whisper_fallback_enables_vad_and_word_timestamps() -> None:
    captured: dict = {}

    class FakeModel:
        def transcribe(self, audio_path: str, **kwargs):
            captured.update(kwargs)
            word = SimpleNamespace(start=0.1, end=0.4, word=" Hello")
            segment = SimpleNamespace(start=0.1, end=1.0, text=" Hello", words=[word])
            info = SimpleNamespace(language="en")
            return iter([segment]), info

    def model_factory(*args, **kwargs):
        captured["model_args"] = args
        captured["model_kwargs"] = kwargs
        return FakeModel()

    document = transcribe_with_whisper(
        "audio.wav",
        video_id="abcdefghijk",
        duration=10,
        model_factory=model_factory,
    )

    assert captured["vad_filter"] is True
    assert captured["word_timestamps"] is True
    assert captured["condition_on_previous_text"] is False
    assert document.segments[0].words[0].text == "Hello"
