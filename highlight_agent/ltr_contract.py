"""Dependency-free LTR feature constants for training-only environments.

Feature extraction keeps its richer contract module under ``features``.  Model
training imports this lightweight copy to avoid importing MediaPipe, OpenCV, or
Hugging Face helpers merely to validate an already-built seven-channel cache.
"""

from __future__ import annotations

LTR_FEATURE_SCHEMA_VERSION = "1.1"
LTR_CHANNEL_ORDER = (
    "rms",
    "pitch",
    "silence",
    "text_score",
    "scene_change",
    "gesture",
    "turn_rate",
)
