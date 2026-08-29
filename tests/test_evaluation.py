"""Unit tests cho module evaluation metrics (Kendall, Spearman, Knapsack, F-score, Hit@K)."""

import numpy as np

from evaluation.metrics import (
    compute_correlation,
    compute_fscore,
    compute_hit_at_k,
    generate_summary_from_scores,
    knapsack_shot_selection,
    temporal_iou,
)


def test_compute_correlation():
    # Điểm hoàn toàn đồng biến
    pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    gt = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

    tau, _ = compute_correlation(pred, gt, method="kendall")
    rho, _ = compute_correlation(pred, gt, method="spearman")

    assert np.isclose(tau, 1.0)
    assert np.isclose(rho, 1.0)


def test_knapsack_shot_selection():
    scores = np.array([10.0, 50.0, 30.0])
    lengths = np.array([5, 10, 5])
    budget = 10

    selected = knapsack_shot_selection(scores, lengths, budget)
    # Chọn shot 1 (score 50, len 10) hoặc shot 0+2 (score 40, len 10) -> shot 1 tối ưu hơn
    assert selected == [1]


def test_compute_fscore():
    # Dự đoán trùng khớp hoàn toàn với 1 user
    pred = np.array([1, 1, 0, 0, 1])
    user = np.array([[1, 1, 0, 0, 1]])

    f1, p, r = compute_fscore(pred, user)
    assert np.isclose(f1, 1.0)
    assert np.isclose(p, 1.0)
    assert np.isclose(r, 1.0)


def test_temporal_iou_and_hit_at_k():
    assert temporal_iou(10, 20, 10, 20) == 1.0
    assert temporal_iou(0, 10, 20, 30) == 0.0
    assert np.isclose(temporal_iou(0, 20, 10, 30), 10.0 / 30.0)

    preds = [
        {"start_time": 10.0, "end_time": 40.0},
        {"start_time": 100.0, "end_time": 130.0},
        {"start_time": 200.0, "end_time": 230.0},
    ]
    gts = [
        {"start_time": 12.0, "end_time": 38.0},  # Match pred 0
        {"start_time": 500.0, "end_time": 530.0},
    ]

    # 1 trong 3 preds trúng -> Hit@3 = 1/3
    hit = compute_hit_at_k(preds, gts, k=3, iou_threshold=0.3)
    assert np.isclose(hit, 1.0 / 3.0)


def test_generate_summary_from_scores():
    frame_scores = np.array([0.1, 0.9, 0.8, 0.2])
    change_points = np.array([[0, 1], [2, 3]])
    picks = np.array([0, 1, 2, 3])

    summary = generate_summary_from_scores(
        frame_scores=frame_scores,
        change_points=change_points,
        n_frames=4,
        n_steps=4,
        picks=picks,
        budget_ratio=0.5,
    )
    assert len(summary) == 4
    assert summary.dtype == np.int32


def test_compute_temporal_precision_recall_f1_and_mean_iou():
    from evaluation.metrics import compute_mean_iou, compute_temporal_precision_recall_f1

    preds = [
        {"start_time": 10.0, "end_time": 40.0},
        {"start_time": 100.0, "end_time": 130.0},
        {"start_time": 200.0, "end_time": 230.0},
    ]
    gts = [
        {"start_time": 10.0, "end_time": 40.0},  # Match pred 0 (IoU = 1.0)
        {"start_time": 105.0, "end_time": 135.0},  # Match pred 1 (IoU = 25/35 = 0.714)
        {"start_time": 400.0, "end_time": 430.0},  # Unmatched
    ]

    p, r, f1 = compute_temporal_precision_recall_f1(preds, gts, k=3, iou_threshold=0.3)
    # TP = 2 (pred 0 and pred 1), top_k=3, len(gts)=3
    assert np.isclose(p, 2.0 / 3.0)
    assert np.isclose(r, 2.0 / 3.0)
    assert np.isclose(f1, 2.0 / 3.0)

    mean_iou = compute_mean_iou(preds, gts, k=3)
    # pred 0 IoU = 1.0, pred 1 IoU = 25/35 ~ 0.714, pred 2 IoU = 0.0 -> mean = (1.0 + 25/35 + 0) / 3
    expected_mean_iou = (1.0 + (25.0 / 35.0) + 0.0) / 3.0
    assert np.isclose(mean_iou, expected_mean_iou)


def test_evaluate_indomain_pipeline(tmp_path):
    import json

    from evaluation.evaluate_indomain import evaluate_indomain

    gt_dir = tmp_path / "ground_truth"
    gt_dir.mkdir()

    sample_gt = {
        "video_id": "test_vid_01",
        "title": "Test Video 01",
        "domain": "lecture",
        "duration": 300.0,
        "highlights": [
            {"highlight_id": "hl_01", "start_time": 10.0, "end_time": 50.0, "importance_score": 5},
            {"highlight_id": "hl_02", "start_time": 100.0, "end_time": 140.0, "importance_score": 4},
        ],
    }
    (gt_dir / "test_vid_01.json").write_text(json.dumps(sample_gt), encoding="utf-8")

    # Evaluate using random baseline
    results = evaluate_indomain(gt_dir=gt_dir, method="random", k=3)
    assert results["evaluated_videos"] == 1
    assert "lecture" in results["domain_summaries"]
    assert "hit@3_iou0.3" in results["overall_summary"]
