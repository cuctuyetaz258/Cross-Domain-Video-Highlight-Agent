"""Đồng bộ bảy nguồn tín hiệu thành feature matrix 10 Hz cho LTR"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

from highlight_agent.features.scoring import normalize_features
from highlight_agent.schemas import AcousticFeatures, FeatureWindow, InteractionFeatures


def _interpolate_windows(
    windows: list[FeatureWindow],
    values: list[float],
    timeline: np.ndarray,
) -> np.ndarray:
    if not windows:
        return np.zeros_like(timeline, dtype=np.float32)
    centers = np.asarray([(window.start + window.end) / 2.0 for window in windows], dtype=np.float32)
    source = np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if len(centers) == 1:
        return np.full_like(timeline, source[0], dtype=np.float32)
    return np.interp(timeline, centers, source).astype(np.float32)


def _text_signal(
    word_scores: list[tuple[float, float, float]],
    *,
    duration: float,
    sample_rate: int,
    sample_count: int,
) -> np.ndarray:
    result = np.zeros(sample_count, dtype=np.float32)
    for start, end, score in word_scores:
        if not np.isfinite(score) or end < start:
            continue
        start_index = max(0, int(np.ceil(start * sample_rate)))
        end_index = min(sample_count, int(np.floor(end * sample_rate)) + 1)
        if start_index < end_index and start < duration:
            result[start_index:end_index] = float(score)
    return result


def _scene_signal(
    scene_times: list[float], *, duration: float, sample_rate: int, sample_count: int) -> np.ndarray:
    result = np.zeros(sample_count, dtype=np.float32)
    for timestamp in scene_times:
        if np.isfinite(timestamp) and 0.0 <= timestamp < duration:
            result[min(sample_count - 1, int(timestamp * sample_rate))] = 1.0
    return gaussian_filter1d(result, sigma=5).astype(np.float32) if sample_count else result


def _gesture_signal(gesture_sparse: np.ndarray, timeline: np.ndarray, duration: float) -> np.ndarray:
    values = np.nan_to_num(np.asarray(gesture_sparse, dtype=np.float32).reshape(-1), nan=0.0)
    if not len(values):
        return np.zeros_like(timeline, dtype=np.float32)
    source_time = np.arange(len(values), dtype=np.float32) / 2.0
    if len(values) == 1:
        return np.full_like(timeline, values[0], dtype=np.float32)
    interpolator = interp1d(source_time, values, kind="linear", fill_value="extrapolate")
    return np.nan_to_num(interpolator(np.minimum(timeline, duration))).astype(np.float32)


def _turn_rate_signal(
    interaction: InteractionFeatures | None,
    acoustic_windows: list[FeatureWindow],
    timeline: np.ndarray,
) -> np.ndarray:
    result = np.zeros_like(timeline, dtype=np.float32)
    if interaction is None:
        return result
    for window in acoustic_windows:
        duration = window.end - window.start
        if duration <= 0:
            continue
        turn_count = sum(window.start <= turn.start < window.end for turn in interaction.turns)
        mask = (timeline >= window.start) & (timeline < window.end)
        result[mask] = turn_count / duration
    return result


def build_feature_matrix(
    acoustic: AcousticFeatures,
    acoustic_windows: list[FeatureWindow],
    scene_times: list[float],
    gesture_sparse: np.ndarray,
    word_scores: list[tuple[float, float, float]],
    interaction: InteractionFeatures | None,
    duration: float,
    sample_rate: int = 10,
) -> np.ndarray:
    """Tạo matrix bảy channel chuẩn hóa có shape 7 x duration nhân sample_rate"""

    if duration <= 0 or sample_rate <= 0:
        raise ValueError("duration and sample_rate must be positive")
    sample_count = int(duration * sample_rate)
    timeline = np.arange(sample_count, dtype=np.float32) / sample_rate
    rms = _interpolate_windows(acoustic_windows, [window.acoustic.rms_mean for window in acoustic_windows], timeline)
    pitch = _interpolate_windows(
        acoustic_windows,
        [window.acoustic.pitch_mean_hz or 0.0 for window in acoustic_windows],
        timeline,
    )
    silence = _interpolate_windows(acoustic_windows, [window.acoustic.silence_ratio for window in acoustic_windows], timeline)
    raw = {
        "rms": rms,
        "pitch": pitch,
        "silence": silence,
        "text_score": _text_signal(word_scores, duration=duration, sample_rate=sample_rate, sample_count=sample_count),
        "scene_change": _scene_signal(scene_times, duration=duration, sample_rate=sample_rate, sample_count=sample_count),
        "gesture": _gesture_signal(gesture_sparse, timeline, duration),
        "turn_rate": _turn_rate_signal(interaction, acoustic_windows, timeline),
    }
    normalized = normalize_features(raw, scaler_type="minmax")
    matrix = np.stack([normalized[name] for name in raw], axis=0)
    return np.clip(np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0).astype(np.float32)
