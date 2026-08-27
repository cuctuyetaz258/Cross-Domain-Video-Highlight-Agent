"""Integration tests for LTR pipeline wiring."""
from __future__ import annotations

import numpy as np
import pytest


def test_state_accepts_ltr_model_path():
    """AgentState TypedDict should accept ltr_model_path=None."""
    from highlight_agent.agent.state import AgentState
    state: AgentState = {  # type: ignore[assignment]
        "video_path": "/tmp/v.mp4",
        "domain": "lecture",
        "ltr_model_path": None,
    }
    assert state["ltr_model_path"] is None


def test_state_accepts_scene_mediapipe():
    """visual_method should accept scene_mediapipe."""
    from highlight_agent.agent.state import AgentState
    state: AgentState = {  # type: ignore[assignment]
        "video_path": "/tmp/v.mp4",
        "domain": "lecture",
        "visual_method": "scene_mediapipe",
    }
    assert state["visual_method"] == "scene_mediapipe"


def test_features_init_exports_new_modules():
    """All new modules should be importable from highlight_agent.features."""
    from highlight_agent.features import (
        build_feature_matrix,
        blend_scores,
        extract_gesture_signal,
        extract_scene_changes,
        extract_topk_nms,
        extract_windows,
    )
    assert callable(build_feature_matrix)
    assert callable(extract_windows)
    assert callable(blend_scores)
    assert callable(extract_topk_nms)
    assert callable(extract_scene_changes)
    assert callable(extract_gesture_signal)


def test_ltr_pipeline_mock_e2e():
    """Run feature matrix through extract_windows -> blend_scores -> extract_topk_nms."""
    import torch
    from highlight_agent.features.sliding_window import extract_windows
    from highlight_agent.features.overlap_blender import blend_scores
    from highlight_agent.features.nms_topk import extract_topk_nms
    from highlight_agent.models.ltr_scorer import AdditiveAttentionScorer

    rng = np.random.default_rng(42)
    feature_matrix = rng.random((7, 600)).astype(np.float32)
    window_tensor = extract_windows(feature_matrix, device="cpu")
    assert window_tensor.shape[1] == 7

    model = AdditiveAttentionScorer()
    model.eval()
    with torch.no_grad():
        raw_scores = model(window_tensor).squeeze(-1)
    window_scores = raw_scores.cpu().numpy()

    T = feature_matrix.shape[1]
    timeline_score = blend_scores(window_scores, T=T)
    assert timeline_score.shape == (T,)

    candidates = extract_topk_nms(timeline_score, k=3, reference_duration=40.0)
    assert isinstance(candidates, list)
    assert len(candidates) <= 3
    for c in candidates:
        assert c.end_time - c.start_time >= 30


def test_analyze_fallback_no_ltr_path():
    """analyze() should not crash when ltr_model_path is None — LTR branch is skipped."""
    from unittest.mock import MagicMock, patch

    from highlight_agent.agent.nodes import analyze
    from highlight_agent.schemas import AcousticFeatures, HighlightCandidate

    mock_acoustic = MagicMock(spec=AcousticFeatures)
    mock_acoustic.duration = 120.0
    mock_acoustic.model_dump.return_value = {}

    mock_workspace = MagicMock()
    mock_workspace.video_id = "test"
    mock_workspace.audio_path = "/dev/null"
    mock_workspace.video_path = "/dev/null"

    mock_transcript = MagicMock()
    mock_transcript.words = []

    state = {
        "video_path": "/dev/null",
        "domain": "lecture",
        "ltr_model_path": None,
        "workspace": mock_workspace,
        "transcript": mock_transcript,
        "highlight_count": 3,
    }

    fake_candidate = HighlightCandidate(
        candidate_id="ltr_01", start_time=0.0, end_time=60.0, score=0.5, reason="test"
    )

    with (
        patch("highlight_agent.agent.nodes.extract_windowed_acoustic_features", return_value=(mock_acoustic, [])),
        patch("highlight_agent.agent.nodes.build_feature_timeline", return_value=MagicMock()),
        patch("highlight_agent.agent.nodes.extract_visual_scores", return_value=[]),
        patch("highlight_agent.agent.nodes.calculate_total_score", return_value=[fake_candidate]),
        patch("highlight_agent.agent.nodes.normalize_features", return_value={}),
        patch("highlight_agent.agent.nodes.save_feature_timeline", return_value="/tmp/f.json"),
        patch("highlight_agent.agent.nodes._naive_candidates", return_value=[]),
    ):
        result = analyze(state)

    assert isinstance(result, dict)
    # LTR branch should NOT have been triggered (no model path)
    assert result.get("features", {}).get("mode") != "ltr_dense_overlap"


def test_analyze_fallback_missing_model_file():
    """analyze() should use old pipeline when model file does not exist."""
    from unittest.mock import MagicMock, patch

    from highlight_agent.agent.nodes import analyze
    from highlight_agent.schemas import AcousticFeatures, HighlightCandidate

    mock_acoustic = MagicMock(spec=AcousticFeatures)
    mock_acoustic.duration = 120.0
    mock_acoustic.model_dump.return_value = {}

    mock_workspace = MagicMock()
    mock_workspace.video_id = "test"
    mock_workspace.audio_path = "/dev/null"
    mock_workspace.video_path = "/dev/null"

    mock_transcript = MagicMock()
    mock_transcript.words = []

    state = {
        "video_path": "/dev/null",
        "domain": "lecture",
        "ltr_model_path": "/nonexistent/path/model.pt",
        "workspace": mock_workspace,
        "transcript": mock_transcript,
        "highlight_count": 3,
    }

    fake_candidate = HighlightCandidate(
        candidate_id="ltr_01", start_time=0.0, end_time=60.0, score=0.5, reason="test"
    )

    with (
        patch("highlight_agent.agent.nodes.extract_windowed_acoustic_features", return_value=(mock_acoustic, [])),
        patch("highlight_agent.agent.nodes.build_feature_timeline", return_value=MagicMock()),
        patch("highlight_agent.agent.nodes.extract_visual_scores", return_value=[]),
        patch("highlight_agent.agent.nodes.calculate_total_score", return_value=[fake_candidate]),
        patch("highlight_agent.agent.nodes.normalize_features", return_value={}),
        patch("highlight_agent.agent.nodes.save_feature_timeline", return_value="/tmp/f.json"),
        patch("highlight_agent.agent.nodes._naive_candidates", return_value=[]),
    ):
        result = analyze(state)

    assert isinstance(result, dict)
    # Model file does not exist → LTR branch skipped
    assert result.get("features", {}).get("mode") != "ltr_dense_overlap"
