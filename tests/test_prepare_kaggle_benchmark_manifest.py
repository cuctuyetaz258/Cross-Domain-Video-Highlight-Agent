from __future__ import annotations

import wave
from pathlib import Path

import pytest

from scripts.prepare_kaggle_benchmark_manifest import adapt_records, normalize_media_id, write_silent_audio


def _record(video_id: str, source: str = "summe") -> dict:
    return {"video_id": video_id, "source": source, "split": "train"}


def test_normalize_media_id_handles_summe_space_underscore_variants() -> None:
    assert normalize_media_id("Base jumping") == "basejumping"
    assert normalize_media_id("Base_jumping") == "basejumping"


def test_adapt_records_resolves_media_and_keeps_labels(tmp_path: Path) -> None:
    media = tmp_path / "public"
    media.mkdir()
    (media / "Base_jumping.mp4").touch()
    records = [{**_record("Base jumping"), "frame_scores": [1.0, 2.0], "fps": 25.0}]

    adapted = adapt_records(records, media, tmp_path / "derived")

    assert adapted[0]["frame_scores"] == [1.0, 2.0]
    assert adapted[0]["video_path"].endswith("Base_jumping.mp4")
    assert adapted[0]["audio_path"].endswith("derived/summe/Base jumping/audio.wav")


def test_adapt_records_rejects_missing_media(tmp_path: Path) -> None:
    media = tmp_path / "public"
    media.mkdir()
    (media / "another-video.mp4").touch()
    with pytest.raises(FileNotFoundError, match="missing 1 benchmark videos"):
        adapt_records([_record("not-present")], media, tmp_path / "derived")


def test_write_silent_audio_preserves_requested_duration(tmp_path: Path) -> None:
    path = tmp_path / "silent.wav"
    write_silent_audio(path, duration=1.25, sample_rate=16_000)

    with wave.open(str(path), "rb") as handle:
        assert handle.getnchannels() == 1
        assert handle.getframerate() == 16_000
        assert handle.getnframes() == 20_000
