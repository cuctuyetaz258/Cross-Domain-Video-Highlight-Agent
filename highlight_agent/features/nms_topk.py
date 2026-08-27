"""Chon cac highlight LTR co diem cao va tach nhau theo NMS"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from highlight_agent.schemas import HighlightCandidate


def _minimum_peak_distance(reference_duration: float, iou_threshold: float, sample_rate: int) -> int:
    """Quy doi khoang cach NMS suy ra tu IoU sang so frame timeline"""
    minimum_seconds = reference_duration * (1 - iou_threshold) / (1 + iou_threshold)
    return round(minimum_seconds * sample_rate)


def extract_topk_nms(
    timeline_score: np.ndarray | object,
    k: int,
    reference_duration: float,
    iou_threshold: float = 0.45,
    sample_rate: int = 10,
) -> list[HighlightCandidate]:
    """Tao toi da k highlight hop le tu cac peak tren score timeline"""
    if k < 0:
        raise ValueError("k khong duoc am")
    if reference_duration <= 0:
        raise ValueError("reference_duration phai lon hon 0")
    if not 0 <= iou_threshold < 1:
        raise ValueError("iou_threshold phai nam trong [0, 1)")
    if sample_rate <= 0:
        raise ValueError("sample_rate phai lon hon 0")

    scores = np.asarray(timeline_score.detach().cpu() if hasattr(timeline_score, "detach") else timeline_score, dtype=np.float32)
    if scores.ndim != 1:
        raise ValueError("timeline_score phai la mang mot chieu")
    if not np.isfinite(scores).all():
        raise ValueError("timeline_score phai huu han")
    if k == 0 or scores.size == 0:
        return []

    peak_indices, _ = find_peaks(scores, height=float(np.mean(scores)))
    minimum_distance = _minimum_peak_distance(reference_duration, iou_threshold, sample_rate)
    ordered_peaks = sorted(peak_indices, key=lambda index: float(scores[index]), reverse=True)
    video_duration = scores.size / sample_rate
    clip_radius = 15 * sample_rate
    selected_peaks: list[int] = []
    candidates: list[HighlightCandidate] = []

    for peak in ordered_peaks:
        if any(abs(peak - selected) < minimum_distance for selected in selected_peaks):
            continue

        start_time = max(0.0, (peak - clip_radius) / sample_rate)
        end_time = min(video_duration, (peak + clip_radius) / sample_rate)
        if end_time - start_time < 30:
            continue

        rank = len(candidates) + 1
        score = float(scores[peak])
        candidates.append(
            HighlightCandidate(
                candidate_id=f"ltr_{rank:02d}",
                start_time=start_time,
                end_time=end_time,
                score=score,
                reason=f"LTR peak tai {peak / sample_rate:.1f}s, score={score:.4f}",
            )
        )
        selected_peaks.append(int(peak))
        if len(candidates) == k:
            break

    return candidates
