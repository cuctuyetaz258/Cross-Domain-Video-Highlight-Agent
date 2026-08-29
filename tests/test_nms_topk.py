import numpy as np
import pytest

from highlight_agent.features.nms_topk import _minimum_peak_distance, extract_topk_nms


def _timeline_with_peaks() -> np.ndarray:
    timeline = np.zeros(1_500, dtype=np.float32)
    timeline[[200, 500, 800, 1_100]] = [0.9, 0.8, 0.7, 0.6]
    return timeline


def test_nms_k_output() -> None:
    candidates = extract_topk_nms(_timeline_with_peaks(), k=3, reference_duration=40)

    assert len(candidates) == 3
    assert [candidate.candidate_id for candidate in candidates] == ["ltr_01", "ltr_02", "ltr_03"]


def test_nms_min_distance() -> None:
    timeline = np.zeros(1_500, dtype=np.float32)
    timeline[[200, 250, 500]] = [0.9, 0.8, 0.7]

    candidates = extract_topk_nms(timeline, k=3, reference_duration=40)
    positions = [round((candidate.start_time + 15) * 10) for candidate in candidates]

    assert len(candidates) == 3
    assert all(
        abs(left - right) >= _minimum_peak_distance(40, 0.45, 10)
        for index, left in enumerate(positions)
        for right in positions[index + 1 :]
    )
    assert "top_score_center" in candidates[-1].reason


def test_nms_candidate_duration() -> None:
    candidates = extract_topk_nms(_timeline_with_peaks(), k=4, reference_duration=40)

    assert all(30 <= candidate.end_time - candidate.start_time <= 90 for candidate in candidates)


def test_nms_iou_derivation() -> None:
    minimum_seconds = 40 * (1 - 0.45) / (1 + 0.45)

    assert minimum_seconds == pytest.approx(15.17, abs=0.01)
    assert _minimum_peak_distance(40, 0.45, 10) == 152


def test_nms_uses_output_clip_duration_when_training_reference_is_short() -> None:
    timeline = np.zeros(2_000, dtype=np.float32)
    timeline[[500, 600, 900]] = [0.9, 0.8, 0.7]

    candidates = extract_topk_nms(timeline, k=3, reference_duration=4.0)

    assert len(candidates) == 3
    for left_index, left in enumerate(candidates):
        for right in candidates[left_index + 1 :]:
            intersection = max(
                0.0,
                min(left.end_time, right.end_time) - max(left.start_time, right.start_time),
            )
            union = (left.end_time - left.start_time) + (right.end_time - right.start_time) - intersection
            assert intersection / union <= 0.45


def test_nms_flat_timeline_relaxes_deterministically() -> None:
    timeline = np.zeros(1_200, dtype=np.float32)

    first = extract_topk_nms(timeline, k=3, reference_duration=40)
    second = extract_topk_nms(timeline, k=3, reference_duration=40)

    assert len(first) == 3
    assert [item.model_dump() for item in first] == [item.model_dump() for item in second]
    assert all("top_score_center" in item.reason for item in first)
