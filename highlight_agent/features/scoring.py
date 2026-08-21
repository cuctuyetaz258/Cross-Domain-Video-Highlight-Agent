"""
Tầng A — Chuẩn hóa đặc trưng & Tính điểm tổng (Feature Normalization & Scoring)

Flow:
  raw features (list[float])
      ↓  normalize_features()
  normalized features ∈ [0, 1]
      ↓  calculate_total_score()
  total score ∈ [0, 1]  (cho mỗi cửa sổ thời gian)

PROFILE_WEIGHTS — bộ trọng số theo domain (lecture, podcast).
Mỗi key là tên tín hiệu, value là trọng số w ∈ [0, 1], tổng = 1.
Trọng số này có thể được thay thế bằng kết quả từ grid_search() để
tìm bộ số tốt nhất dựa trên dữ liệu chuẩn.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import product
from typing import Literal

import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Bộ trọng số mặc định theo domain
# Nguồn: đề cương final project (Table 2.1)
# Sẽ được thay thế bởi grid_search() khi có ground truth.
# ──────────────────────────────────────────────

Domain = Literal["lecture", "podcast"]

SIGNAL_NAMES = ("semantic", "acoustic", "interaction", "visual")

PROFILE_WEIGHTS: dict[Domain, dict[str, float]] = {
    "lecture": {
        "semantic":    0.50,
        "acoustic":    0.30,
        "interaction": 0.00,
        "visual":      0.20,
    },
    "podcast": {
        "semantic":    0.30,
        "acoustic":    0.20,
        "interaction": 0.30,
        "visual":      0.20,
    },
}


# ──────────────────────────────────────────────
# Data class kết quả scoring
# ──────────────────────────────────────────────

@dataclass
class WindowScore:
    """Kết quả đầy đủ cho một cửa sổ thời gian sau khi tính điểm."""

    window_idx: int
    start: float
    end: float
    total_score: float                           # ∈ [0, 1]
    signals_raw: dict[str, float]                # trước chuẩn hóa
    signals_normalized: dict[str, float]         # sau chuẩn hóa ∈ [0, 1]
    weights: dict[str, float]                    # trọng số đã dùng
    extra: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.end - self.start

    def summary(self) -> str:
        sigs = ", ".join(
            f"{k}={v:.3f}" for k, v in self.signals_normalized.items()
        )
        return (
            f"[{self.start:.1f}s–{self.end:.1f}s] "
            f"total={self.total_score:.3f} | {sigs}"
        )


# ──────────────────────────────────────────────
# Phần A.1 — Chuẩn hóa đặc trưng
# ──────────────────────────────────────────────

ScalerType = Literal["minmax", "robust"]


def normalize_features(
    features: dict[str, list[float]],
    scaler_type: ScalerType = "minmax",
    clip: bool = True,
) -> dict[str, np.ndarray]:
    """
    Chuẩn hóa từng tín hiệu (mảng 1-D) về khoảng [0, 1].

    Args:
        features   : Dict ánh xạ tên tín hiệu → list điểm raw, một phần tử
                     mỗi cửa sổ. Tất cả list phải có cùng độ dài.
                     Ví dụ:
                       {
                         "semantic":    [0.2, 0.8, 0.5, ...],
                         "acoustic":    [10.3, 22.1, 5.0, ...],
                         "interaction": [0.0, 1.0, 0.5, ...],
                         "visual":      [3.2, 50.1, 8.4, ...],
                       }
        scaler_type: "minmax" — MinMaxScaler (tốt khi không có outlier mạnh).
                     "robust" — RobustScaler (tốt khi RMS energy có đột biến).
        clip       : Nếu True, clip kết quả về [0, 1] để tránh float lỗi nhỏ.

    Returns:
        Dict tên tín hiệu → numpy array đã chuẩn hóa ∈ [0, 1].

    Raises:
        ValueError: Nếu các mảng không cùng độ dài hoặc rỗng.

    Example:
        >>> raw = {"semantic": [0.1, 0.9, 0.5], "acoustic": [5.0, 30.0, 12.0]}
        >>> normed = normalize_features(raw)
        >>> normed["semantic"]   # array([0., 1., 0.5])
        >>> normed["acoustic"]   # array([0., 1., 0.28...])
    """
    if not features:
        raise ValueError("features dict rỗng — không có gì để chuẩn hóa.")

    # Kiểm tra độ dài nhất quán
    lengths = {k: len(v) for k, v in features.items()}
    unique_lengths = set(lengths.values())
    if len(unique_lengths) > 1:
        raise ValueError(
            f"Các mảng features có độ dài khác nhau: {lengths}. "
            "Tất cả phải có cùng số cửa sổ."
        )
    n_windows = next(iter(unique_lengths))
    if n_windows == 0:
        raise ValueError("Các mảng features rỗng (0 cửa sổ).")

    # Chọn scaler
    if scaler_type == "minmax":
        scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    elif scaler_type == "robust":
        # RobustScaler dùng median + IQR, sau đó clip về [0,1]
        scaler = RobustScaler()
    else:
        raise ValueError(f"scaler_type không hợp lệ: {scaler_type!r}. Chọn 'minmax' hoặc 'robust'.")

    normalized: dict[str, np.ndarray] = {}

    for signal_name, raw_values in features.items():
        arr = np.array(raw_values, dtype=np.float64).reshape(-1, 1)

        # Xử lý trường hợp tất cả giá trị giống nhau (tránh chia 0)
        if np.all(arr == arr[0]):
            logger.warning(
                "Tín hiệu '%s' có tất cả giá trị bằng nhau (%.4f). "
                "Sẽ trả về mảng 0.0.",
                signal_name,
                float(arr.flat[0]),
            )
            normalized[signal_name] = np.zeros(n_windows, dtype=np.float64)
            continue

        scaled = scaler.fit_transform(arr).flatten()

        # Nếu dùng RobustScaler, cần clip tay về [0, 1]
        if clip or scaler_type == "robust":
            scaled = np.clip(scaled, 0.0, 1.0)

        normalized[signal_name] = scaled
        logger.debug(
            "Normalize '%s': raw=[%.3f, %.3f] → scaled=[%.3f, %.3f]",
            signal_name,
            float(arr.min()),
            float(arr.max()),
            float(scaled.min()),
            float(scaled.max()),
        )

    return normalized


# ──────────────────────────────────────────────
# Phần A.2 — Tính điểm tổng
# ──────────────────────────────────────────────

def calculate_total_score(
    normalized_features: dict[str, np.ndarray],
    weights: dict[str, float],
    window_starts: list[float] | None = None,
    window_ends: list[float] | None = None,
) -> list[WindowScore]:
    """
    Tính điểm tổng có trọng số cho từng cửa sổ và trả về danh sách WindowScore.

    Công thức:
        Score(t) = Σ  w_i × f_i(t)        (với Σ w_i = 1)

    Args:
        normalized_features: Kết quả từ normalize_features() — giá trị ∈ [0, 1].
        weights            : Dict tên tín hiệu → trọng số. Tổng phải = 1.0 (± 0.01).
                             Tín hiệu không có trong normalized_features sẽ bị bỏ qua.
        window_starts      : List timestamp bắt đầu mỗi cửa sổ (giây). Mặc định None.
        window_ends        : List timestamp kết thúc mỗi cửa sổ (giây). Mặc định None.

    Returns:
        List[WindowScore] đã sắp xếp theo thứ tự cửa sổ (không phải theo điểm).
        Dùng sorted(..., key=lambda x: x.total_score, reverse=True) để ranking.

    Raises:
        ValueError: Nếu tổng trọng số sai hoặc độ dài arrays không khớp.

    Example:
        >>> normed = normalize_features(raw)
        >>> scores = calculate_total_score(normed, PROFILE_WEIGHTS["lecture"])
        >>> top3 = sorted(scores, key=lambda s: s.total_score, reverse=True)[:3]
    """
    # Kiểm tra trọng số
    active_weights = {k: v for k, v in weights.items() if k in normalized_features}
    if not active_weights:
        raise ValueError(
            "Không có tín hiệu nào trong weights khớp với normalized_features. "
            f"weights keys: {list(weights.keys())}, "
            f"features keys: {list(normalized_features.keys())}"
        )

    weight_sum = sum(active_weights.values())
    if not (0.99 <= weight_sum <= 1.01):
        logger.warning(
            "Tổng trọng số active = %.4f (khác 1.0). "
            "Sẽ tự động normalize lại trọng số.",
            weight_sum,
        )
        active_weights = {k: v / weight_sum for k, v in active_weights.items()}

    # Số cửa sổ
    n_windows = len(next(iter(normalized_features.values())))

    # Tạo timestamps mặc định nếu không có
    if window_starts is None:
        window_starts = list(range(n_windows))      # 0, 1, 2, ...
    if window_ends is None:
        window_ends = [s + 1 for s in window_starts]

    if len(window_starts) != n_windows or len(window_ends) != n_windows:
        raise ValueError(
            f"window_starts/ends có độ dài {len(window_starts)}/{len(window_ends)} "
            f"nhưng features có {n_windows} cửa sổ."
        )

    results: list[WindowScore] = []

    for idx in range(n_windows):
        signals_raw = {k: float(normalized_features[k][idx]) for k in active_weights}
        total = sum(active_weights[k] * signals_raw[k] for k in active_weights)
        total = float(np.clip(total, 0.0, 1.0))

        results.append(
            WindowScore(
                window_idx=idx,
                start=float(window_starts[idx]),
                end=float(window_ends[idx]),
                total_score=round(total, 6),
                signals_raw=signals_raw,           # alias normalized để rõ nghĩa
                signals_normalized=signals_raw,
                weights=dict(active_weights),
            )
        )

    return results


def score_from_domain(
    normalized_features: dict[str, np.ndarray],
    domain: Domain,
    window_starts: list[float] | None = None,
    window_ends: list[float] | None = None,
) -> list[WindowScore]:
    """
    Shortcut: tính điểm dùng PROFILE_WEIGHTS mặc định của domain.

    Args:
        normalized_features: Kết quả từ normalize_features().
        domain             : "lecture" | "podcast".

    Returns:
        List[WindowScore] theo thứ tự cửa sổ.
    """
    if domain not in PROFILE_WEIGHTS:
        raise ValueError(f"Domain không hỗ trợ: {domain!r}. Chọn {list(PROFILE_WEIGHTS.keys())}")
    return calculate_total_score(
        normalized_features,
        PROFILE_WEIGHTS[domain],
        window_starts=window_starts,
        window_ends=window_ends,
    )


# ──────────────────────────────────────────────
# Phần A.3 — Grid Search trọng số
# ──────────────────────────────────────────────

@dataclass
class GridSearchResult:
    """Kết quả Grid Search tốt nhất."""

    best_weights: dict[str, float]
    best_metric: float
    metric_name: str
    all_results: list[tuple[dict[str, float], float]] = field(default_factory=list)

    def summary(self) -> str:
        w_str = ", ".join(f"{k}={v:.2f}" for k, v in self.best_weights.items())
        return f"Best {self.metric_name}={self.best_metric:.4f} | weights: {{{w_str}}}"


def _f1_score_highlight(
    scores: list[WindowScore],
    ground_truth_windows: set[int],
    top_k: int = 3,
) -> float:
    """
    Tính F1 đơn giản: so sánh top-K windows được chọn với ground truth.

    Args:
        scores              : Danh sách WindowScore.
        ground_truth_windows: Set các window_idx được đánh dấu là highlight.
        top_k               : Số windows chọn để so sánh.

    Returns:
        F1-score ∈ [0, 1].
    """
    ranked = sorted(scores, key=lambda s: s.total_score, reverse=True)
    predicted = {s.window_idx for s in ranked[:top_k]}

    tp = len(predicted & ground_truth_windows)
    precision = tp / top_k if top_k > 0 else 0.0
    recall = tp / len(ground_truth_windows) if ground_truth_windows else 0.0

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def grid_search_weights(
    normalized_features: dict[str, np.ndarray],
    ground_truth_windows: set[int],
    signal_names: tuple[str, ...] = SIGNAL_NAMES,
    weight_step: float = 0.1,
    top_k: int = 3,
    metric: Literal["f1"] = "f1",
) -> GridSearchResult:
    """
    Duyệt qua tổ hợp trọng số và tìm bộ cho metric cao nhất.

    Ràng buộc: tổng tất cả trọng số = 1.0, mỗi trọng số ∈ [0.0, 1.0].

    Args:
        normalized_features : Kết quả normalize_features() — array ∈ [0, 1].
        ground_truth_windows: Set window_idx được gán nhãn là highlight.
        signal_names        : Tuple tên các tín hiệu cần tìm trọng số.
                              Chỉ tính với các tín hiệu có trong normalized_features.
        weight_step         : Bước nhảy của trọng số (0.1 → 11 giá trị mỗi tín hiệu).
                              Nhỏ hơn = chính xác hơn nhưng chậm hơn (O(step^n)).
        top_k               : Số windows lấy để tính metric.
        metric              : Hàm đánh giá. Hiện tại chỉ hỗ trợ "f1".

    Returns:
        GridSearchResult chứa best_weights và toàn bộ kết quả.

    Complexity:
        Với 4 tín hiệu, step=0.1 → ~286 tổ hợp hợp lệ (rất nhanh).
        Với step=0.05 → ~1771 tổ hợp.

    Example:
        >>> result = grid_search_weights(normed, ground_truth={2, 5, 8})
        >>> print(result.summary())
        Best f1=0.8000 | weights: {semantic=0.50, acoustic=0.30, ...}
    """
    # Lọc chỉ những signal có trong features
    active_signals = [s for s in signal_names if s in normalized_features]
    if not active_signals:
        raise ValueError(
            f"Không có signal nào trong {signal_names} khớp với features. "
            f"Available: {list(normalized_features.keys())}"
        )

    logger.info(
        "Grid Search bắt đầu: %d signals, step=%.2f, top_k=%d, metric=%s",
        len(active_signals),
        weight_step,
        top_k,
        metric,
    )

    weight_values = np.round(np.arange(0.0, 1.0 + weight_step, weight_step), 10)
    n_signals = len(active_signals)
    tol = weight_step / 2  # Dung sai so sánh tổng trọng số

    best_metric_value = -1.0
    best_weights: dict[str, float] = {}
    all_results: list[tuple[dict[str, float], float]] = []

    combo_count = 0
    valid_count = 0

    for combo in product(weight_values, repeat=n_signals):
        combo_count += 1
        total = sum(combo)
        # Chỉ giữ tổ hợp có tổng ≈ 1.0
        if abs(total - 1.0) > tol:
            continue
        valid_count += 1

        weights = {active_signals[i]: float(combo[i]) for i in range(n_signals)}
        window_scores = calculate_total_score(
            normalized_features,
            weights,
        )

        if metric == "f1":
            metric_value = _f1_score_highlight(window_scores, ground_truth_windows, top_k)
        else:
            raise ValueError(f"metric không hỗ trợ: {metric!r}")

        all_results.append((weights, metric_value))

        if metric_value > best_metric_value:
            best_metric_value = metric_value
            best_weights = dict(weights)
            logger.debug(
                "Cải thiện: %s=%.4f | weights=%s",
                metric,
                metric_value,
                weights,
            )

    logger.info(
        "Grid Search xong: %d tổ hợp kiểm tra (%d hợp lệ), best_%s=%.4f",
        combo_count,
        valid_count,
        metric,
        best_metric_value,
    )

    return GridSearchResult(
        best_weights=best_weights,
        best_metric=best_metric_value,
        metric_name=metric,
        all_results=all_results,
    )
