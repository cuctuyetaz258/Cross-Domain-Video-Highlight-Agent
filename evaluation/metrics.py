"""Các hàm tính toán Metric chuẩn cho Benchmark (TVSum, SumMe) và Dataset In-Domain."""

import numpy as np
from scipy.stats import kendalltau, spearmanr

# =====================================================================
# 1. RANKING & CORRELATION METRICS (Dùng cho TVSum Importance Scoring)
# =====================================================================

def compute_correlation(
    pred_scores: np.ndarray,
    gt_scores: np.ndarray,
    method: str = "kendall",
) -> tuple[float, float]:
    """
    Tính hệ số tương quan thứ hạng giữa điểm dự đoán và điểm chuẩn con người chấm.
    - 'kendall': Kendall's tau-b (τ)
    - 'spearman': Spearman's rho (ρ)
    """
    pred = np.asarray(pred_scores, dtype=np.float64).flatten()
    gt = np.asarray(gt_scores, dtype=np.float64).flatten()

    if len(pred) != len(gt):
        raise ValueError(f"Độ dài không khớp: pred={len(pred)}, gt={len(gt)}")

    # Tránh lỗi nếu mảng hằng số (không có độ biến thiên)
    if np.all(pred == pred[0]) or np.all(gt == gt[0]):
        return 0.0, 1.0

    if method == "kendall":
        res = kendalltau(pred, gt)
        tau = float(res.statistic) if hasattr(res, "statistic") else float(res[0])
        p_val = float(res.pvalue) if hasattr(res, "pvalue") else float(res[1])
        return (0.0 if np.isnan(tau) else tau), p_val
    elif method == "spearman":
        res = spearmanr(pred, gt)
        rho = float(res.statistic) if hasattr(res, "statistic") else float(res[0])
        p_val = float(res.pvalue) if hasattr(res, "pvalue") else float(res[1])
        return (0.0 if np.isnan(rho) else rho), p_val
    else:
        raise ValueError(f"Không hỗ trợ method='{method}'. Chọn 'kendall' hoặc 'spearman'.")


# =====================================================================
# 2. VIDEO SUMMARIZATION F-SCORE (Dùng cho SumMe & TVSum Overlap Protocol)
# =====================================================================

def knapsack_shot_selection(
    shot_scores: np.ndarray,
    shot_lengths: np.ndarray,
    max_budget: int,
) -> list[int]:
    """
    Thuật toán quy hoạch động 0/1 Knapsack chọn các cảnh (shots) có tổng điểm cao nhất
    nhưng tổng thời lượng không vượt quá ngân sách (thường là 15% độ dài video).
    """
    n = len(shot_scores)
    # Quy đổi về integer cho Knapsack
    weights = [int(length) for length in shot_lengths]
    values = [float(score) * 1000 for score in shot_scores]
    capacity = int(max_budget)

    dp = np.zeros((n + 1, capacity + 1), dtype=np.float64)

    for i in range(1, n + 1):
        w = weights[i - 1]
        v = values[i - 1]
        for c in range(capacity + 1):
            if w <= c:
                dp[i, c] = max(dp[i - 1, c], dp[i - 1, c - w] + v)
            else:
                dp[i, c] = dp[i - 1, c]

    # Truy vết chọn cảnh
    selected = []
    c = capacity
    for i in range(n, 0, -1):
        if dp[i, c] != dp[i - 1, c]:
            selected.append(i - 1)
            c -= weights[i - 1]

    return sorted(selected)


def generate_summary_from_scores(
    frame_scores: np.ndarray,
    change_points: np.ndarray,
    n_frames: int,
    n_steps: int,
    picks: np.ndarray,
    budget_ratio: float = 0.15,
) -> np.ndarray:
    """
    Từ điểm số từng frame -> tính điểm trung bình cho từng shot (dựa vào change_points)
    -> dùng Knapsack chọn Top shots dưới ngân sách 15% -> tạo mảng nhị phân 0/1 (Summary).
    """
    # 1. Tính điểm trung bình cho từng shot (segment)
    shot_scores = []
    shot_lengths = []

    # Map điểm từ n_steps (frame được sample) về scale thực tế
    picks = np.asarray(picks, dtype=np.int32)
    step_scores = np.asarray(frame_scores, dtype=np.float64)

    for shot_idx, (start, end) in enumerate(change_points):
        # Lấy các frame thuộc shot này
        mask = (picks >= start) & (picks <= end)
        if np.any(mask):
            score = float(np.mean(step_scores[mask]))
        else:
            score = 0.0
        shot_scores.append(score)
        shot_lengths.append(end - start + 1)

    shot_scores = np.array(shot_scores, dtype=np.float64)
    shot_lengths = np.array(shot_lengths, dtype=np.int32)

    # 2. Chọn cảnh theo ngân sách
    max_budget = int(n_frames * budget_ratio)
    selected_shots = knapsack_shot_selection(shot_scores, shot_lengths, max_budget)

    # 3. Tạo vector nhị phân 0/1 cho toàn bộ n_frames
    summary = np.zeros(n_frames, dtype=np.int32)
    for shot_idx in selected_shots:
        start, end = change_points[shot_idx]
        summary[start : min(end + 1, n_frames)] = 1

    return summary


def compute_fscore(
    pred_summary: np.ndarray,
    user_summaries: np.ndarray,
    eval_metric: str = "mean",
) -> tuple[float, float, float]:
    """
    Tính F1-score, Precision, Recall giữa summary dự đoán và user summaries của nhiều người gán nhãn.
    - TVSum chuẩn: lấy 'mean' qua tất cả users.
    - SumMe chuẩn: lấy 'max' qua tất cả users (hoặc 'mean').
    """
    pred = np.asarray(pred_summary, dtype=np.int32).flatten()
    user_summaries = np.asarray(user_summaries, dtype=np.int32)

    if user_summaries.ndim == 1:
        user_summaries = user_summaries[np.newaxis, :]

    # Cắt ngắn nếu có lệch nhẹ về frame count
    min_len = min(len(pred), user_summaries.shape[1])
    pred = pred[:min_len]
    user_summaries = user_summaries[:, :min_len]

    f_scores = []
    precisions = []
    recalls = []

    for user in user_summaries:
        overlap = np.sum(pred * user)
        p = overlap / np.sum(pred) if np.sum(pred) > 0 else 0.0
        r = overlap / np.sum(user) if np.sum(user) > 0 else 0.0
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0

        f_scores.append(f1)
        precisions.append(p)
        recalls.append(r)

    if eval_metric == "max":
        best_idx = np.argmax(f_scores)
        return f_scores[best_idx], precisions[best_idx], recalls[best_idx]
    else:  # 'mean'
        return float(np.mean(f_scores)), float(np.mean(precisions)), float(np.mean(recalls))


# =====================================================================
# 3. TEMPORAL MATCHING & HIT@K (Dùng cho In-Domain Dataset)
# =====================================================================

def temporal_iou(
    pred_start: float,
    pred_end: float,
    gt_start: float,
    gt_end: float,
) -> float:
    """Tính Intersection over Union giữa 2 đoạn thời gian (giây)."""
    inter_start = max(pred_start, gt_start)
    inter_end = min(pred_end, gt_end)
    intersection = max(0.0, inter_end - inter_start)

    union = (pred_end - pred_start) + (gt_end - gt_start) - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def compute_hit_at_k(
    predictions: list[dict],
    ground_truths: list[dict],
    k: int = 3,
    iou_threshold: float = 0.3,
) -> float:
    """
    Hit@K: Tỷ lệ clip trong Top K dự đoán có Temporal IoU >= iou_threshold với ít nhất 1 clip chuẩn.
    """
    top_k = predictions[:k]
    if not top_k:
        return 0.0

    hits = 0
    matched_gt = set()

    for pred in top_k:
        p_start = float(pred.get("start_time", 0))
        p_end = float(pred.get("end_time", 0))

        for gt_idx, gt in enumerate(ground_truths):
            if gt_idx in matched_gt:
                continue
            g_start = float(gt.get("start_time", 0))
            g_end = float(gt.get("end_time", 0))

            if temporal_iou(p_start, p_end, g_start, g_end) >= iou_threshold:
                hits += 1
                matched_gt.add(gt_idx)
                break

    return hits / k


def compute_temporal_precision_recall_f1(
    predictions: list[dict],
    ground_truths: list[dict],
    k: int = 3,
    iou_threshold: float = 0.3,
) -> tuple[float, float, float]:
    """
    Tính Precision@K, Recall@K và F1@K dựa trên Temporal IoU matching.
    - True Positive (TP): số lượng clip trong Top-K dự đoán match 1-1 với một ground truth (IoU >= iou_threshold).
    - Precision = TP / len(top_k)
    - Recall = TP / len(ground_truths)
    - F1 = 2 * Precision * Recall / (Precision + Recall)
    """
    top_k = predictions[:k]
    if not top_k or not ground_truths:
        return 0.0, 0.0, 0.0

    matched_gt = set()
    tp = 0

    for pred in top_k:
        p_start = float(pred.get("start_time", 0))
        p_end = float(pred.get("end_time", 0))

        best_iou = 0.0
        best_gt_idx = -1
        for gt_idx, gt in enumerate(ground_truths):
            if gt_idx in matched_gt:
                continue
            g_start = float(gt.get("start_time", 0))
            g_end = float(gt.get("end_time", 0))
            iou = temporal_iou(p_start, p_end, g_start, g_end)
            if iou >= iou_threshold and iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_gt_idx != -1:
            tp += 1
            matched_gt.add(best_gt_idx)

    precision = tp / len(top_k) if top_k else 0.0
    recall = tp / len(ground_truths) if ground_truths else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1


def compute_mean_iou(
    predictions: list[dict],
    ground_truths: list[dict],
    k: int = 3,
) -> float:
    """
    Tính Mean IoU: trung bình của IoU cao nhất với ground truth cho mỗi clip trong Top-K dự đoán.
    """
    top_k = predictions[:k]
    if not top_k or not ground_truths:
        return 0.0

    ious = []
    for pred in top_k:
        p_start = float(pred.get("start_time", 0))
        p_end = float(pred.get("end_time", 0))

        max_iou = 0.0
        for gt in ground_truths:
            g_start = float(gt.get("start_time", 0))
            g_end = float(gt.get("end_time", 0))
            iou = temporal_iou(p_start, p_end, g_start, g_end)
            if iou > max_iou:
                max_iou = iou
        ious.append(max_iou)

    return float(np.mean(ious)) if ious else 0.0

