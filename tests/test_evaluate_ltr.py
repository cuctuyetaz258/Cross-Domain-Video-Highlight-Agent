from __future__ import annotations

import numpy as np
import pytest

from evaluation.evaluate_ltr import profile_weight_scores, window_metrics
from highlight_agent.models.train_offline import WindowExample


def _example(index: int, label: int, score: float) -> WindowExample:
    return WindowExample(
        video_id="video",
        domain="benchmark",
        window_index=index,
        start=float(index),
        end=float(index + 1),
        feature=np.zeros(7, dtype=np.float32),
        label=label,
        score=score,
    )


def test_profile_weight_scores_maps_canonical_channels() -> None:
    features = np.zeros((2, 7), dtype=np.float32)
    features[0, 3] = 1.0  # semantic
    features[1, 4] = 1.0  # scene change

    scores = profile_weight_scores(features, "lecture")

    # Constant silence=0 contributes 0.3 to the acoustic aggregate.
    assert scores[0] == pytest.approx(0.50 + 0.30 * 0.30)
    assert scores[1] == pytest.approx(0.20 * 0.50 + 0.30 * 0.30)


def test_window_metrics_reports_perfect_ranking() -> None:
    examples = [
        _example(0, 0, 0.1),
        _example(1, 0, 0.2),
        _example(2, 1, 0.8),
        _example(3, 1, 0.9),
    ]

    metrics = window_metrics(examples, np.asarray([0.1, 0.2, 0.8, 0.9]), top_k=2)

    assert metrics["average_precision"] == pytest.approx(1.0)
    assert metrics["window_f1_at_positive_count"] == pytest.approx(1.0)
    assert metrics["positive_hit_at_k"] == pytest.approx(1.0)
    assert metrics["kendall_tau"] == pytest.approx(1.0)
    assert metrics["spearman_rho"] == pytest.approx(1.0)


def test_window_metrics_rejects_mismatched_score_count() -> None:
    with pytest.raises(ValueError, match="counts do not match"):
        window_metrics([_example(0, 1, 1.0)], np.asarray([]), top_k=1)
