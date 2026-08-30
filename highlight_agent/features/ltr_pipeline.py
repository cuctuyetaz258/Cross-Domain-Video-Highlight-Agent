"""One feature extraction path shared by runtime inference and offline caches."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from highlight_agent.schemas import AcousticFeatures, InteractionFeatures, TranscriptDocument

from .acoustic import extract_windowed_acoustic_features
from .alignment import build_feature_matrix
from .interaction import extract_interaction_features
from .ltr_contract import (
    LTR_CHANNEL_ORDER,
    LTR_SAMPLE_RATE,
    LTRPipelineError,
    feature_contract,
)
from .semantic import transcript_tfidf_density_scores
from .visual_new import extract_gesture_observation, extract_scene_observation

Domain = Literal["lecture", "podcast", "standup", "benchmark"]


@dataclass(frozen=True)
class LTRFeatureBundle:
    """Canonical LTR matrix plus provenance needed for debugging and parity."""

    matrix: np.ndarray
    acoustic: AcousticFeatures
    acoustic_windows: list[Any]
    interaction: InteractionFeatures | None
    metadata: dict[str, Any]


def _validate_matrix(matrix: np.ndarray) -> None:
    if matrix.dtype != np.float32:
        raise LTRPipelineError("LTR_FEATURE_SCHEMA_MISMATCH", f"dtype={matrix.dtype}; expected float32")
    if matrix.ndim != 2 or matrix.shape[0] != len(LTR_CHANNEL_ORDER) or matrix.shape[1] == 0:
        raise LTRPipelineError(
            "LTR_FEATURE_SCHEMA_MISMATCH",
            f"shape={matrix.shape}; expected ({len(LTR_CHANNEL_ORDER)}, T) with T > 0",
        )
    if not np.isfinite(matrix).all():
        raise LTRPipelineError("LTR_FEATURE_NON_FINITE", "feature matrix contains NaN or Inf")
    if float(matrix.min()) < -1e-6 or float(matrix.max()) > 1.0 + 1e-6:
        raise LTRPipelineError("LTR_FEATURE_SCHEMA_MISMATCH", "feature values must be in [0, 1]")


def build_ltr_features(
    *,
    video_path: str | Path,
    audio_path: str | Path,
    transcript: TranscriptDocument,
    domain: Domain,
    duration: float,
    known_speaker_count: int | None = None,
    min_speaker_count: int | None = None,
    max_speaker_count: int | None = None,
    include_scenes: bool = True,
    include_gesture: bool = True,
    gesture_sample_rate: float = 0.2,
    device: str = "cpu",
) -> LTRFeatureBundle:
    """Build the production seven-channel, 10 Hz matrix exactly once."""

    if duration < 5.0:
        raise LTRPipelineError("LTR_VIDEO_TOO_SHORT", "video must be at least 5 seconds")
    stage_seconds: dict[str, float] = {}

    started = time.perf_counter()
    try:
        acoustic, acoustic_windows = extract_windowed_acoustic_features(audio_path)
    except Exception as exc:
        raise LTRPipelineError("LTR_AUDIO_EXTRACTION_FAILED", str(exc)) from exc
    stage_seconds["acoustic"] = time.perf_counter() - started
    if abs(float(acoustic.duration) - duration) > 2.0:
        raise LTRPipelineError(
            "LTR_MEDIA_DURATION_MISMATCH",
            f"audio duration {acoustic.duration:.3f}s differs from video duration {duration:.3f}s",
        )

    started = time.perf_counter()
    word_scores = transcript_tfidf_density_scores(transcript)
    stage_seconds["semantic"] = time.perf_counter() - started

    started = time.perf_counter()
    try:
        scene = (
            extract_scene_observation(video_path, duration)
            if include_scenes
            else None
        )
    except Exception as exc:
        raise LTRPipelineError("LTR_SCENE_EXTRACTION_FAILED", str(exc)) from exc
    stage_seconds["scene"] = time.perf_counter() - started
    if scene is not None and scene.status == "extraction_failed":
        raise LTRPipelineError(
            "LTR_SCENE_EXTRACTION_FAILED",
            scene.error or "SceneDetect failed without details",
        )

    started = time.perf_counter()
    try:
        gesture = (
            extract_gesture_observation(
                video_path,
                duration,
                sample_rate=gesture_sample_rate,
            )
            if include_gesture
            else None
        )
    except Exception as exc:
        raise LTRPipelineError("LTR_GESTURE_EXTRACTION_FAILED", str(exc)) from exc
    stage_seconds["gesture"] = time.perf_counter() - started
    if gesture is not None and gesture.status in {
        "video_unreadable",
        "facemesh_initialization_failed",
        "no_decodable_samples",
        "no_samples",
    }:
        raise LTRPipelineError(
            "LTR_GESTURE_EXTRACTION_FAILED",
            f"MediaPipe gesture status={gesture.status}: {gesture.error or 'no details'}",
        )

    interaction = None
    started = time.perf_counter()
    if domain == "podcast":
        try:
            interaction = extract_interaction_features(
                audio_path,
                num_speakers=known_speaker_count,
                min_speakers=min_speaker_count,
                max_speakers=max_speaker_count,
                duration=acoustic.duration,
            )
        except Exception as exc:
            raise LTRPipelineError("LTR_INTERACTION_EXTRACTION_FAILED", str(exc)) from exc
    stage_seconds["interaction"] = time.perf_counter() - started

    scene_times = scene.timestamps if scene is not None else []
    gesture_signal = (
        gesture.signal
        if gesture is not None
        else np.zeros(int(duration * 2.0), dtype=np.float32)
    )
    matrix = build_feature_matrix(
        acoustic,
        acoustic_windows,
        scene_times,
        gesture_signal,
        word_scores,
        interaction,
        duration,
        sample_rate=LTR_SAMPLE_RATE,
    )
    _validate_matrix(matrix)

    channel_stats = {
        name: {
            "min": float(values.min()),
            "max": float(values.max()),
            "mean": float(values.mean()),
            "std": float(values.std()),
            "zero_ratio": float(np.mean(values == 0.0)),
        }
        for name, values in zip(LTR_CHANNEL_ORDER, matrix)
    }
    metadata = {
        "feature_contract": feature_contract(),
        "shape": list(matrix.shape),
        "dtype": str(matrix.dtype),
        "duration": duration,
        "domain": domain,
        "extractor": {
            "device": device,
            "text_method": "segment_tfidf_density",
            "scene_enabled": include_scenes,
            "scene_status": scene.status if scene is not None else "disabled",
            "gesture_enabled": include_gesture,
            "gesture_sample_rate": gesture_sample_rate,
            "gesture_status": gesture.status if gesture is not None else "disabled",
            "interaction_method": "pyannote" if domain == "podcast" else "zero_non_podcast",
        },
        "observations": {
            "acoustic_window_count": len(acoustic_windows),
            "text_interval_count": len(word_scores),
            "scene_count": len(scene_times),
            "gesture_sample_count": int(len(gesture_signal)),
            "gesture_decoded_sample_count": gesture.decoded_sample_count if gesture else 0,
            "gesture_detected_sample_count": gesture.detected_sample_count if gesture else 0,
            "speaker_turn_count": interaction.turn_count if interaction else 0,
        },
        "channel_stats": channel_stats,
        "stage_seconds": {key: round(value, 3) for key, value in stage_seconds.items()},
    }
    return LTRFeatureBundle(matrix, acoustic, acoustic_windows, interaction, metadata)
