import wave

import pytest

from scripts.prepare_summe import (
    _pad_wav_to_duration,
    _probe_fps,
    _write_silent_audio,
    assign_summe_splits,
)


def test_assign_summe_splits_is_deterministic_and_video_level():
    records = [
        {"video_id": f"video-{index}", "source": "summe", "domain": "benchmark"}
        for index in range(10)
    ]

    first = assign_summe_splits(
        records, train_count=6, val_count=2, test_count=2, seed=42
    )
    second = assign_summe_splits(
        records, train_count=6, val_count=2, test_count=2, seed=42
    )

    assert first == second
    assert sum(record["split"] == "train" for record in first) == 6
    assert sum(record["split"] == "val" for record in first) == 2
    assert sum(record["split"] == "test" for record in first) == 2
    assert {record["domain"] for record in first} == {"benchmark"}


def test_write_silent_audio_has_requested_duration(tmp_path):
    output = tmp_path / "silent.wav"

    _write_silent_audio(output, 1.25)

    with wave.open(str(output), "rb") as handle:
        assert handle.getnchannels() == 1
        assert handle.getframerate() == 16_000
        assert handle.getnframes() == 20_000


def test_pad_wav_to_duration_appends_silence(tmp_path):
    output = tmp_path / "short.wav"
    _write_silent_audio(output, 0.5)

    padded_seconds = _pad_wav_to_duration(output, 1.25)

    assert padded_seconds == pytest.approx(0.75)
    with wave.open(str(output), "rb") as handle:
        assert handle.getnframes() == 20_000


def test_pad_wav_to_duration_does_not_truncate_longer_audio(tmp_path):
    output = tmp_path / "long.wav"
    _write_silent_audio(output, 1.25)

    assert _pad_wav_to_duration(output, 0.5) == 0.0
    with wave.open(str(output), "rb") as handle:
        assert handle.getnframes() == 20_000


def test_probe_fps_rejects_unreadable_media(tmp_path):
    invalid = tmp_path / "invalid.mp4"
    invalid.write_bytes(b"not a video")

    with pytest.raises(ValueError, match="cannot determine FPS"):
        _probe_fps(invalid)
