import wave
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("librosa", reason="librosa không có — skip acoustic tests")

from highlight_agent.features import (
    extract_acoustic_features,
    extract_windowed_acoustic_features,
)


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int = 16_000) -> None:
    clipped = np.clip(samples, -1, 1)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes((clipped * 32767).astype("<i2").tobytes())


def test_acoustic_extractor_reports_energy_pitch_and_silence(tmp_path: Path) -> None:
    sample_rate = 16_000
    tone_time = np.arange(sample_rate * 2) / sample_rate
    tone = 0.5 * np.sin(2 * np.pi * 220 * tone_time)
    samples = np.concatenate([tone, np.zeros(int(0.6 * sample_rate)), tone[:sample_rate]])
    audio_path = tmp_path / "tone-with-silence.wav"
    _write_wav(audio_path, samples, sample_rate)

    features = extract_acoustic_features(audio_path)

    assert features.duration == 3.6
    assert features.rms_peak > 0.3
    assert features.pitch_mean_hz is not None
    assert 210 < features.pitch_mean_hz < 230
    assert features.voiced_ratio > 0.5
    assert features.silence_duration >= 0.4
    assert any(interval.start < 2.3 < interval.end for interval in features.silence_intervals)


def test_acoustic_extractor_handles_silent_audio(tmp_path: Path) -> None:
    audio_path = tmp_path / "silent.wav"
    _write_wav(audio_path, np.zeros(16_000))

    features = extract_acoustic_features(audio_path)

    assert features.rms_mean == 0
    assert features.pitch_mean_hz is None
    assert features.voiced_ratio == 0
    assert features.silence_ratio == 1


def test_windowed_acoustic_extractor_keeps_partial_final_window(tmp_path: Path) -> None:
    audio_path = tmp_path / "short-tone.wav"
    samples = 0.5 * np.sin(2 * np.pi * 220 * np.arange(16_000 * 3.6) / 16_000)
    _write_wav(audio_path, samples)

    global_features, windows = extract_windowed_acoustic_features(audio_path)

    assert global_features.duration == 3.6
    assert len(windows) == 1
    assert windows[0].start == 0
    assert windows[0].end == 3.6
    assert windows[0].acoustic.duration == 3.6
