"""Chon cac highlight LTR co diem cao va tach nhau theo NMS"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from highlight_agent.schemas import HighlightCandidate

OUTPUT_CLIP_DURATION_SECONDS = 30.0


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

    scores = np.asarray(
        timeline_score.detach().cpu() if hasattr(timeline_score, "detach") else timeline_score, dtype=np.float32
    )
    if scores.ndim != 1:
        raise ValueError("timeline_score phai la mang mot chieu")
    if not np.isfinite(scores).all():
        raise ValueError("timeline_score phai huu han")
    if k == 0 or scores.size == 0:
        return []

    strict_peaks, _ = find_peaks(scores, height=float(np.mean(scores)))
    local_peaks, _ = find_peaks(scores)
    # Training labels can be much shorter than the clips emitted below (TVSum
    # currently yields L_ref ~= 4 s while runtime clips are 30 s).  NMS must use
    # at least the actual output duration or the resulting clips can violate its
    # own IoU threshold even when their score peaks passed suppression.
    nms_reference_duration = max(reference_duration, OUTPUT_CLIP_DURATION_SECONDS)
    minimum_distance = _minimum_peak_distance(
        nms_reference_duration,
        iou_threshold,
        sample_rate,
    )
    strict_order = sorted(strict_peaks, key=lambda index: (-float(scores[index]), int(index)))
    local_order = sorted(local_peaks, key=lambda index: (-float(scores[index]), int(index)))
    all_order = sorted(range(scores.size), key=lambda index: (-float(scores[index]), int(index)))
    video_duration = scores.size / sample_rate
    clip_radius = round(OUTPUT_CLIP_DURATION_SECONDS * sample_rate / 2)
    selected_centers: list[int] = []
    candidates: list[HighlightCandidate] = []

    # Normalize to [0, 1] so HighlightCandidate score validator (>= 0) always passes
    s_min, s_max = scores.min(), scores.max()
    score_range = s_max - s_min
    norm_scores = (scores - s_min) / score_range if score_range > 0 else np.zeros_like(scores)

    if video_duration < OUTPUT_CLIP_DURATION_SECONDS:
        return []

    seen: set[int] = set()
    passes = (
        ("above_mean_peak", strict_order),
        ("local_peak", local_order),
        ("top_score_center", all_order),
    )
    for stage, (strategy, ordered_peaks) in enumerate(passes, start=1):
        for peak in ordered_peaks:
            peak = int(peak)
            if peak in seen:
                continue
            seen.add(peak)
            centered_start = (peak - clip_radius) / sample_rate
            start_time = min(
                max(0.0, centered_start),
                video_duration - OUTPUT_CLIP_DURATION_SECONDS,
            )
            start_time = round(start_time, 3)
            end_time = round(start_time + OUTPUT_CLIP_DURATION_SECONDS, 3)
            effective_center = round(
                (start_time + OUTPUT_CLIP_DURATION_SECONDS / 2) * sample_rate
            )
            if any(
                abs(effective_center - selected) < minimum_distance
                for selected in selected_centers
            ):
                continue

            rank = len(candidates) + 1
            score = float(norm_scores[peak])
            candidates.append(
                HighlightCandidate(
                    candidate_id=f"ltr_{rank:02d}",
                    start_time=start_time,
                    end_time=end_time,
                    score=score,
                    reason=(
                        f"LTR {strategy} at {peak / sample_rate:.1f}s, "
                        f"normalized_score={score:.4f}"
                    ),
                    signals={"ltr_score": score, "relaxation_stage": float(stage)},
                )
            )
            selected_centers.append(effective_center)
            if len(candidates) == k:
                return candidates

    return candidates
