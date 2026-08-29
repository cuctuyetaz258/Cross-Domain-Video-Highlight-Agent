from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from highlight_agent.features.ltr_contract import LTRPipelineError, feature_contract
from highlight_agent.features.visual_new import GestureExtraction, SceneExtraction


def _patch_common(monkeypatch, *, scene: SceneExtraction, gesture: GestureExtraction) -> np.ndarray:
    from highlight_agent.features import ltr_pipeline

    acoustic = MagicMock()
    acoustic.duration = 60.0
    expected = np.linspace(0, 1, 7 * 600, dtype=np.float32).reshape(7, 600)
    monkeypatch.setattr(
        ltr_pipeline,
        "extract_windowed_acoustic_features",
        lambda path: (acoustic, []),
    )
    monkeypatch.setattr(ltr_pipeline, "transcript_tfidf_density_scores", lambda transcript: [])
    monkeypatch.setattr(ltr_pipeline, "extract_scene_observation", lambda *args: scene)
    monkeypatch.setattr(ltr_pipeline, "extract_gesture_observation", lambda *args, **kwargs: gesture)
    monkeypatch.setattr(ltr_pipeline, "build_feature_matrix", lambda *args, **kwargs: expected.copy())
    return expected


def test_unified_extractor_accepts_no_face_and_is_deterministic(monkeypatch) -> None:
    from highlight_agent.features.ltr_pipeline import build_ltr_features

    expected = _patch_common(
        monkeypatch,
        scene=SceneExtraction([], "no_scene_detected"),
        gesture=GestureExtraction(
            np.zeros(120, dtype=np.float32),
            "no_face_detected",
            120,
            0,
        ),
    )
    kwargs = {
        "video_path": "video.mp4",
        "audio_path": "audio.wav",
        "transcript": MagicMock(),
        "domain": "lecture",
        "duration": 60.0,
    }

    runtime = build_ltr_features(**kwargs)
    offline = build_ltr_features(**kwargs)

    np.testing.assert_array_equal(runtime.matrix, expected)
    np.testing.assert_array_equal(runtime.matrix, offline.matrix)
    assert runtime.metadata["feature_contract"] == feature_contract()
    assert runtime.metadata["extractor"]["scene_status"] == "no_scene_detected"
    assert runtime.metadata["extractor"]["gesture_status"] == "no_face_detected"


@pytest.mark.parametrize(
    ("scene", "gesture", "error_code"),
    [
        (
            SceneExtraction([], "extraction_failed", "decoder error"),
            GestureExtraction(np.zeros(120, dtype=np.float32), "ok", 120, 120),
            "LTR_SCENE_EXTRACTION_FAILED",
        ),
        (
            SceneExtraction([], "no_scene_detected"),
            GestureExtraction(
                np.zeros(120, dtype=np.float32),
                "facemesh_initialization_failed",
                0,
                0,
            ),
            "LTR_GESTURE_EXTRACTION_FAILED",
        ),
    ],
)
def test_unified_extractor_preserves_technical_failures(
    monkeypatch,
    scene: SceneExtraction,
    gesture: GestureExtraction,
    error_code: str,
) -> None:
    from highlight_agent.features.ltr_pipeline import build_ltr_features

    _patch_common(monkeypatch, scene=scene, gesture=gesture)

    with pytest.raises(LTRPipelineError, match=error_code):
        build_ltr_features(
            video_path="video.mp4",
            audio_path="audio.wav",
            transcript=MagicMock(),
            domain="lecture",
            duration=60.0,
        )
