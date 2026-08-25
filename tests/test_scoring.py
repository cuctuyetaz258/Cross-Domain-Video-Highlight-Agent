"""
Unit tests cho scoring.py (Feature Normalization & Multi-Signal Scoring).
Chạy bằng: pytest tests/test_scoring.py -v
"""

import math

import numpy as np
import pytest

from highlight_agent.features.scoring import (
    PROFILE_WEIGHTS,
    GridSearchResult,
    WindowScore,
    calculate_total_score,
    grid_search_weights,
    normalize_features,
    score_from_domain,
)


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

@pytest.fixture()
def simple_raw() -> dict[str, list[float]]:
    """4 cửa sổ, 4 tín hiệu."""
    return {
        "semantic":    [0.1, 0.9, 0.5, 0.3],
        "acoustic":    [5.0, 30.0, 12.0, 8.0],
        "interaction": [0.0, 1.0, 0.5, 0.2],
        "visual":      [2.0, 80.0, 20.0, 5.0],
    }


@pytest.fixture()
def normed(simple_raw) -> dict[str, np.ndarray]:
    return normalize_features(simple_raw)


# ──────────────────────────────────────────────
# normalize_features
# ──────────────────────────────────────────────

class TestNormalizeFeatures:
    def test_output_range_minmax(self, normed):
        """Mọi giá trị sau normalize phải ∈ [0, 1]."""
        for signal_name, arr in normed.items():
            assert arr.min() >= 0.0 - 1e-9, f"{signal_name}: min < 0"
            assert arr.max() <= 1.0 + 1e-9, f"{signal_name}: max > 1"

    def test_min_is_zero_max_is_one(self, normed):
        """MinMaxScaler phải map min→0, max→1."""
        for signal_name, arr in normed.items():
            assert math.isclose(arr.min(), 0.0, abs_tol=1e-6), f"{signal_name}: min != 0"
            assert math.isclose(arr.max(), 1.0, abs_tol=1e-6), f"{signal_name}: max != 1"

    def test_shape_preserved(self, simple_raw, normed):
        """Số phần tử không thay đổi."""
        for k in simple_raw:
            assert len(normed[k]) == len(simple_raw[k])

    def test_constant_signal_returns_zeros(self):
        """Khi tất cả giá trị bằng nhau → trả về mảng 0."""
        raw = {"semantic": [5.0, 5.0, 5.0]}
        result = normalize_features(raw)
        assert np.allclose(result["semantic"], 0.0)

    def test_robust_scaler_still_clips(self):
        """RobustScaler có outlier lớn vẫn phải clip về [0, 1]."""
        raw = {"acoustic": [1.0, 2.0, 3.0, 4.0, 1000.0]}
        result = normalize_features(raw, scaler_type="robust")
        assert result["acoustic"].min() >= 0.0
        assert result["acoustic"].max() <= 1.0

    def test_raises_on_empty(self):
        with pytest.raises(ValueError, match="rỗng"):
            normalize_features({})

    def test_raises_on_length_mismatch(self):
        with pytest.raises(ValueError, match="độ dài|số cửa sổ"):
            normalize_features({"semantic": [1.0, 2.0], "acoustic": [3.0]})

    def test_raises_on_invalid_scaler(self):
        with pytest.raises(ValueError, match="scaler_type"):
            normalize_features({"x": [1.0, 2.0]}, scaler_type="invalid")  # type: ignore


# ──────────────────────────────────────────────
# calculate_total_score
# ──────────────────────────────────────────────

class TestCalculateTotalScore:
    def test_returns_correct_count(self, normed):
        scores = calculate_total_score(normed, PROFILE_WEIGHTS["lecture"])
        assert len(scores) == 4

    def test_score_in_range(self, normed):
        scores = calculate_total_score(normed, PROFILE_WEIGHTS["lecture"])
        for s in scores:
            assert 0.0 <= s.total_score <= 1.0, f"score out of range: {s.total_score}"

    def test_highest_score_correct_window(self, normed):
        """Window index 1 có raw cao nhất ở mọi tín hiệu → phải đứng đầu."""
        scores = calculate_total_score(normed, PROFILE_WEIGHTS["lecture"])
        best = max(scores, key=lambda s: s.total_score)
        assert best.window_idx == 1

    def test_weight_auto_normalize_when_sum_not_one(self, normed):
        """Khi tổng trọng số != 1, hàm phải tự chuẩn hóa và không raise."""
        weights = {"semantic": 0.5, "acoustic": 0.5, "visual": 0.5}  # tổng = 1.5
        scores = calculate_total_score(normed, weights)
        for s in scores:
            assert 0.0 <= s.total_score <= 1.0

    def test_raises_when_no_matching_signals(self, normed):
        with pytest.raises(ValueError, match="Không có tín hiệu"):
            calculate_total_score(normed, {"nonexistent": 1.0})

    def test_window_timestamps_attached(self, normed):
        starts = [0.0, 30.0, 60.0, 90.0]
        ends   = [30.0, 60.0, 90.0, 120.0]
        scores = calculate_total_score(normed, PROFILE_WEIGHTS["lecture"], starts, ends)
        for i, s in enumerate(scores):
            assert s.start == starts[i]
            assert s.end == ends[i]

    def test_signals_normalized_matches_input(self, normed):
        """signals_normalized dict phải có đúng keys từ active weights."""
        scores = calculate_total_score(normed, PROFILE_WEIGHTS["lecture"])
        active = {k for k, v in PROFILE_WEIGHTS["lecture"].items() if v > 0}
        for s in scores:
            assert set(s.signals_normalized.keys()) == set(s.weights.keys())


# ──────────────────────────────────────────────
# score_from_domain
# ──────────────────────────────────────────────

class TestScoreFromDomain:
    def test_lecture_domain(self, normed):
        scores = score_from_domain(normed, "lecture")
        assert len(scores) == 4

    def test_podcast_domain(self, normed):
        scores = score_from_domain(normed, "podcast")
        assert len(scores) == 4

    def test_standup_domain(self, normed):
        scores = score_from_domain(normed, "standup")
        assert len(scores) == 4

    def test_invalid_domain_raises(self, normed):
        with pytest.raises(ValueError, match="Domain"):
            score_from_domain(normed, "invalid_domain")  # type: ignore


# ──────────────────────────────────────────────
# grid_search_weights
# ──────────────────────────────────────────────

class TestGridSearchWeights:
    def test_returns_valid_result(self, normed):
        result = grid_search_weights(normed, ground_truth_windows={1}, top_k=1, weight_step=0.5)
        assert isinstance(result, GridSearchResult)
        assert result.best_metric >= 0.0

    def test_best_weights_sum_to_one(self, normed):
        result = grid_search_weights(normed, ground_truth_windows={1}, top_k=1, weight_step=0.5)
        total = sum(result.best_weights.values())
        assert math.isclose(total, 1.0, abs_tol=0.02)

    def test_best_metric_is_max(self, normed):
        result = grid_search_weights(normed, ground_truth_windows={1}, top_k=1, weight_step=0.5)
        all_metrics = [m for _, m in result.all_results]
        assert math.isclose(result.best_metric, max(all_metrics), abs_tol=1e-9)

    def test_perfect_f1_when_window1_is_ground_truth(self, normed):
        """
        Window 1 có điểm cao nhất với mọi tín hiệu.
        Nếu ground_truth = {1}, top_k=1, F1 tối đa phải đạt được.
        """
        result = grid_search_weights(normed, ground_truth_windows={1}, top_k=1, weight_step=0.5)
        assert result.best_metric == pytest.approx(1.0, abs=1e-6)

    def test_summary_string(self, normed):
        result = grid_search_weights(normed, ground_truth_windows={1}, top_k=1, weight_step=0.5)
        s = result.summary()
        assert "Best" in s
        assert "weights" in s

    def test_raises_when_no_matching_signals(self, normed):
        with pytest.raises(ValueError, match="Không có signal"):
            grid_search_weights(normed, ground_truth_windows={1}, signal_names=("xyz",))
