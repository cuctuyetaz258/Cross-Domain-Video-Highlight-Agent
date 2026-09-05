import pytest

from evaluation.evaluate_actionformer_ltr import detection_average_precision
from highlight_agent.models.actionformer import TemporalProposal


def test_detection_average_precision_is_one_for_correct_ranking() -> None:
    targets = {"video": [TemporalProposal(10, 40, 1.0, -1, 0)]}
    predictions = [
        ("video", TemporalProposal(10, 40, 0.9, 0, 1)),
        ("video", TemporalProposal(60, 90, 0.1, 0, 2)),
    ]

    assert detection_average_precision(predictions, targets, iou_threshold=0.5) == pytest.approx(1.0)


def test_detection_average_precision_penalizes_false_positive_ranking() -> None:
    targets = {"video": [TemporalProposal(10, 40, 1.0, -1, 0)]}
    predictions = [
        ("video", TemporalProposal(60, 90, 0.9, 0, 1)),
        ("video", TemporalProposal(10, 40, 0.8, 0, 2)),
    ]

    assert detection_average_precision(predictions, targets, iou_threshold=0.5) == pytest.approx(0.5)
