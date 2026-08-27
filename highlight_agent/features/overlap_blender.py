"""Tron diem cua cac cua so LTR chong lap thanh timeline 10 Hz"""

from __future__ import annotations

import numpy as np


def blend_scores(
    window_scores: np.ndarray | object,
    T: int,
    window_size: int = 50,
    hop_size: int = 10,
) -> np.ndarray:
    """Cong va lay trung binh diem cua moi cua so tren timeline"""
    if T < 0:
        raise ValueError("T khong duoc am")
    if window_size <= 0 or hop_size <= 0:
        raise ValueError("window_size va hop_size phai lon hon 0")

    # asarray ho tro ca Tensor CPU ma khong can phu thuoc torch trong module nay
    scores = np.asarray(window_scores.detach().cpu() if hasattr(window_scores, "detach") else window_scores, dtype=np.float32)
    if scores.ndim != 1:
        raise ValueError("window_scores phai la mang mot chieu")

    timeline_score = np.zeros(T, dtype=np.float32)
    overlap_count = np.zeros(T, dtype=np.float32)
    for index, score in enumerate(scores):
        start = index * hop_size
        if start >= T:
            break
        end = min(start + window_size, T)
        timeline_score[start:end] += score
        overlap_count[start:end] += 1.0

    valid = overlap_count > 0
    timeline_score[valid] /= overlap_count[valid]
    return timeline_score
