"""Trích xuất các features RMS, pitch và silence"""

import os
import tempfile
from pathlib import Path

# Numba cần thư mục cache có thể ghi trước khi Librosa được import
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "highlight-agent-numba-cache"))

try:
    import librosa
except ImportError:
    librosa = None
import numpy as np

from highlight_agent.schemas import AcousticFeatures, FeatureWindow, TimeInterval

DEFAULT_FRAME_LENGTH = 2048
DEFAULT_HOP_LENGTH = 512
DEFAULT_FMIN_HZ = 50.0
DEFAULT_FMAX_HZ = 400.0
DEFAULT_SILENCE_THRESHOLD_DB = 40.0
DEFAULT_MIN_SILENCE_DURATION = 0.30


def _complement_intervals(
    non_silent_intervals: np.ndarray,
    *,
    sample_count: int,
    sample_rate: int,
    min_duration: float,
) -> list[TimeInterval]:
    """Chuyển các khoảng có tiếng thành các khoảng im lặng đã lọc"""

    silence: list[TimeInterval] = []
    cursor = 0
    for start, end in non_silent_intervals:
        start = int(max(cursor, start))
        end = int(min(sample_count, end))
        if start > cursor:
            interval = TimeInterval(start=cursor / sample_rate, end=start / sample_rate)
            if interval.end - interval.start >= min_duration:
                silence.append(interval)
        cursor = max(cursor, end)

    if cursor < sample_count:
        interval = TimeInterval(start=cursor / sample_rate, end=sample_count / sample_rate)
        if interval.end - interval.start >= min_duration:
            silence.append(interval)
    return silence


def _pitch_statistics(
    samples: np.ndarray,
    *,
    sample_rate: int,
    frame_length: int,
    hop_length: int,
    fmin_hz: float,
    fmax_hz: float,
) -> tuple[dict[str, float | None], float]:
    # YIN keeps the same F0 range as the former pYIN implementation but avoids
    # pYIN's large candidate-probability matrix on multi-minute recordings.
    f0 = librosa.yin(
        samples,
        fmin=fmin_hz,
        fmax=fmax_hz,
        sr=sample_rate,
        frame_length=frame_length,
        hop_length=hop_length,
    )
    voiced = f0[np.isfinite(f0)]
    if len(voiced):
        median = float(np.median(voiced))
        # Suppress octave/silence outliers that YIN can emit without pYIN's
        # voiced-probability post-processing.
        voiced = voiced[(voiced >= median * 0.75) & (voiced <= median * 1.25)]
    voiced_ratio = float(np.mean(np.isfinite(f0))) if len(f0) else 0.0
    if not len(voiced):
        return {
            "pitch_mean_hz": None,
            "pitch_median_hz": None,
            "pitch_std_hz": None,
            "pitch_min_hz": None,
            "pitch_max_hz": None,
        }, voiced_ratio
    return {
        "pitch_mean_hz": float(np.mean(voiced)),
        "pitch_median_hz": float(np.median(voiced)),
        "pitch_std_hz": float(np.std(voiced)),
        "pitch_min_hz": float(np.min(voiced)),
        "pitch_max_hz": float(np.max(voiced)),
    }, voiced_ratio


def _extract_from_samples(
    samples: np.ndarray,
    *,
    sample_rate: int,
    frame_length: int,
    hop_length: int,
    fmin_hz: float,
    fmax_hz: float,
    silence_threshold_db: float,
    min_silence_duration: float,
    include_pitch: bool = True,
) -> AcousticFeatures:
    if sample_rate <= 0 or samples.size == 0:
        raise ValueError("audio must contain at least one sample")

    duration = float(samples.size / sample_rate)
    rms = librosa.feature.rms(
        y=samples,
        frame_length=frame_length,
        hop_length=hop_length,
    )[0]
    if float(np.max(rms)) <= np.finfo(rms.dtype).eps:
        silence_intervals = [TimeInterval(start=0.0, end=duration)]
        silence_duration = duration
        pitch = {
            "pitch_mean_hz": None,
            "pitch_median_hz": None,
            "pitch_std_hz": None,
            "pitch_min_hz": None,
            "pitch_max_hz": None,
        }
        voiced_ratio = 0.0
    else:
        non_silent = librosa.effects.split(
            samples,
            top_db=silence_threshold_db,
            frame_length=frame_length,
            hop_length=hop_length,
        )
        silence_intervals = _complement_intervals(
            non_silent,
            sample_count=samples.size,
            sample_rate=sample_rate,
            min_duration=min_silence_duration,
        )
        silence_duration = float(sum(interval.end - interval.start for interval in silence_intervals))
        if include_pitch:
            pitch, voiced_ratio = _pitch_statistics(
                samples,
                sample_rate=sample_rate,
                frame_length=frame_length,
                hop_length=hop_length,
                fmin_hz=fmin_hz,
                fmax_hz=fmax_hz,
            )
        else:
            pitch = {
                "pitch_mean_hz": None,
                "pitch_median_hz": None,
                "pitch_std_hz": None,
                "pitch_min_hz": None,
                "pitch_max_hz": None,
            }
            voiced_ratio = 0.0

    return AcousticFeatures(
        duration=duration,
        rms_mean=float(np.mean(rms)),
        rms_peak=float(np.max(rms)),
        rms_p95=float(np.percentile(rms, 95)),
        rms_std=float(np.std(rms)),
        voiced_ratio=voiced_ratio,
        silence_duration=silence_duration,
        silence_ratio=silence_duration / duration,
        silence_intervals=silence_intervals,
        **pitch,
    )


def extract_acoustic_features(
    audio_path: str | Path,
    *,
    frame_length: int = DEFAULT_FRAME_LENGTH,
    hop_length: int = DEFAULT_HOP_LENGTH,
    fmin_hz: float = DEFAULT_FMIN_HZ,
    fmax_hz: float = DEFAULT_FMAX_HZ,
    silence_threshold_db: float = DEFAULT_SILENCE_THRESHOLD_DB,
    min_silence_duration: float = DEFAULT_MIN_SILENCE_DURATION,
) -> AcousticFeatures:
    """Trích xuất thống kê âm học trên toàn bộ tệp audio

    Backend tạo WAV mono 16 kHz, nhưng Librosa vẫn đọc sample rate thật
    để hàm hoạt động đúng với các tệp mẫu trong test
    """

    if frame_length <= 0 or hop_length <= 0:
        raise ValueError("frame_length and hop_length must be positive")
    if not 0 < fmin_hz < fmax_hz:
        raise ValueError("pitch limits must satisfy 0 < fmin_hz < fmax_hz")
    if silence_threshold_db <= 0 or min_silence_duration < 0:
        raise ValueError("silence threshold and minimum duration must be positive")

    if librosa is None:
        raise ImportError("Vui lòng cài đặt librosa: pip install librosa")
    samples, sample_rate = librosa.load(Path(audio_path), sr=None, mono=True)
    return _extract_from_samples(
        samples,
        sample_rate=sample_rate,
        frame_length=frame_length,
        hop_length=hop_length,
        fmin_hz=fmin_hz,
        fmax_hz=fmax_hz,
        silence_threshold_db=silence_threshold_db,
        min_silence_duration=min_silence_duration,
    )


def extract_windowed_acoustic_features(
    audio_path: str | Path,
    *,
    window_seconds: float = 30.0,
    hop_seconds: float = 30.0,
    frame_length: int = DEFAULT_FRAME_LENGTH,
    hop_length: int = DEFAULT_HOP_LENGTH,
    fmin_hz: float = DEFAULT_FMIN_HZ,
    fmax_hz: float = DEFAULT_FMAX_HZ,
    silence_threshold_db: float = DEFAULT_SILENCE_THRESHOLD_DB,
    min_silence_duration: float = DEFAULT_MIN_SILENCE_DURATION,
) -> tuple[AcousticFeatures, list[FeatureWindow]]:
    """Trích xuất feature âm học toàn cục và từng cửa sổ 30 giây trong một lần đọc audio"""

    if window_seconds <= 0 or hop_seconds <= 0:
        raise ValueError("window_seconds and hop_seconds must be positive")
    if librosa is None:
        raise ImportError("Vui lòng cài đặt librosa: pip install librosa")
    samples, sample_rate = librosa.load(Path(audio_path), sr=None, mono=True)
    global_features = _extract_from_samples(
        samples,
        sample_rate=sample_rate,
        frame_length=frame_length,
        hop_length=hop_length,
        fmin_hz=fmin_hz,
        fmax_hz=fmax_hz,
        silence_threshold_db=silence_threshold_db,
        min_silence_duration=min_silence_duration,
        # Full-file pYIN allocates a very large candidate matrix on long
        # recordings. The LTR matrix consumes the per-window pitch below.
        include_pitch=False,
    )

    windows: list[FeatureWindow] = []
    start = 0.0
    while start < global_features.duration:
        end = min(start + window_seconds, global_features.duration)
        start_index = round(start * sample_rate)
        end_index = round(end * sample_rate)
        window_features = _extract_from_samples(
            samples[start_index:end_index],
            sample_rate=sample_rate,
            frame_length=frame_length,
            hop_length=hop_length,
            fmin_hz=fmin_hz,
            fmax_hz=fmax_hz,
            silence_threshold_db=silence_threshold_db,
            min_silence_duration=min_silence_duration,
        )
        windows.append(FeatureWindow(start=start, end=end, acoustic=window_features))
        start += hop_seconds
    return global_features, windows
