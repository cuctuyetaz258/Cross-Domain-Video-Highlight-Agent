"""
Tầng A — Chuẩn hóa đặc trưng & Tính điểm tổng (Feature Normalization & Scoring)

Flow:
  raw features (dict[str, list[float]])
      ↓  normalize_features()
  normalized features ∈ [0, 1]
      ↓  calculate_total_score()
  total score ∈ [0, 1]  (cho mỗi cửa sổ thời gian)

PROFILE_WEIGHTS — bộ trọng số theo domain (lecture, podcast, standup).
Mỗi key là tên tín hiệu ("semantic", "acoustic", "interaction", "visual"), value là trọng số w ∈ [0, 1], tổng = 1.
Trọng số này có thể được thay thế bằng kết quả từ grid_search_weights() để
tìm bộ số tốt nhất dựa trên ground-truth data.
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

Domain = Literal["lecture", "podcast", "standup"]

SIGNAL_NAMES = ("semantic", "acoustic", "interaction", "visual")

PROFILE_WEIGHTS: dict[Domain, dict[str, float]] = {
    "lecture": {
        "semantic": 0.50,
        "acoustic": 0.30,
        "interaction": 0.00,
        "visual": 0.20,
    },
    "podcast": {
        "semantic": 0.30,
        "acoustic": 0.20,
        "interaction": 0.30,
        "visual": 0.20,
    },
    "standup": {
        "semantic": 0.30,
        "acoustic": 0.35,
        "interaction": 0.15,
        "visual": 0.20,
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
    features: dict[str, list[float] | np.ndarray],
    scaler_type: ScalerType = "minmax",
    clip: bool = True,
) -> dict[str, np.ndarray]:
    """
    Chuẩn hóa từng tín hiệu (mảng 1-D) về khoảng [0, 1].

    Args:
        features   : Dict {tên_tín_hiệu: mảng_giá_trị_thô_theo_từng_cửa_sổ}.
                     Tất cả các mảng phải có cùng độ dài (số cửa sổ).
        scaler_type: "minmax"  — MinMaxScaler, nhạy với outlier.
                     "robust"  — RobustScaler (dùng median và IQR), chống nhiễu tốt hơn.
        clip       : Nếu True, ép giá trị về [0, 1] sau khi chuẩn hóa.

    Returns:
        Dict {tên_tín_hiệu: numpy array 1-D ∈ [0, 1]}.

    Example:
        >>> raw = {"semantic": [0.1, 0.9, 0.5], "acoustic": [5.0, 30.0, 12.0]}
        >>> normed = normalize_features(raw)
        >>> normed["semantic"]   # array([0., 1., 0.5])
    """
    if not features:
        raise ValueError("features dict không được rỗng.")

    # Kiểm tra tính đồng nhất về độ dài
    lengths = {k: len(v) for k, v in features.items()}
    unique_lengths = set(lengths.values())
    if len(unique_lengths) > 1:
        raise ValueError(
            f"Các tín hiệu không cùng số cửa sổ: {lengths}. "
            "Tất cả tín hiệu phải có cùng độ dài."
        )

    n_samples = next(iter(unique_lengths))
    if n_samples == 0:
        return {k: np.array([], dtype=np.float32) for k in features}

    normalized: dict[str, np.ndarray] = {}

    for name, raw_vals in features.items():
        arr = np.array(raw_vals, dtype=np.float32).reshape(-1, 1)

        # Trường hợp mảng hằng (tất cả giá trị bằng nhau)
        if np.all(arr == arr[0]):
            logger.debug("Tín hiệu %r có giá trị không đổi (%s) → trả về 0.0", name, arr[0, 0])
            normalized[name] = np.zeros(n_samples, dtype=np.float32)
            continue

        if scaler_type == "minmax":
            scaler = MinMaxScaler(feature_range=(0.0, 1.0))
            normed = scaler.fit_transform(arr).ravel()
        elif scaler_type == "robust":
            scaler = RobustScaler()
            scaled = scaler.fit_transform(arr).ravel()
            # RobustScaler không cố định min/max → MinMax tiếp để về [0, 1]
            s_min, s_max = scaled.min(), scaled.max()
            if s_max > s_min:
                normed = (scaled - s_min) / (s_max - s_min)
            else:
                normed = np.zeros_like(scaled)
        else:
            raise ValueError(f"scaler_type không hợp lệ: {scaler_type!r}. Chọn 'minmax' hoặc 'robust'.")

        if clip:
            normed = np.clip(normed, 0.0, 1.0)

        normalized[name] = normed.astype(np.float32)

    return normalized


# ──────────────────────────────────────────────
# Phần A.2 — Tính điểm tổng hợp (Weighted Fusion)
# ──────────────────────────────────────────────

def calculate_total_score(
    normalized_features: dict[str, np.ndarray],
    weights: dict[str, float],
    window_starts: list[float] | None = None,
    window_ends: list[float] | None = None,
) -> list[WindowScore]:
    """
    Tính điểm tổng hợp cho từng cửa sổ: total = sum(w_i * signal_i).

    Args:
        normalized_features: Kết quả từ normalize_features() — mỗi value là array ∈ [0, 1].
        weights            : Dict {tên_tín_hiệu: trọng_số}.
        window_starts      : Danh sách giây bắt đầu của từng cửa sổ (tùy chọn).
        window_ends        : Danh sách giây kết thúc của từng cửa sổ (tùy chọn).

    Returns:
        List[WindowScore] theo thứ tự thời gian.
    """
    if not normalized_features:
        return []

    # Lọc các trọng số có tín hiệu tương ứng
    active_weights = {
        k: float(v)
        for k, v in weights.items()
        if k in normalized_features and v > 0
    }

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
        window_starts = [float(i) for i in range(n_windows)]
    if window_ends is None:
        window_ends = [s + 1.0 for s in window_starts]

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
                signals_raw=signals_raw,
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
        domain             : "lecture" | "podcast" | "standup".

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
    Tính F1: so sánh top-K windows được chọn với ground truth.

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
        top_k               : Số windows lấy để tính metric.
        metric              : Hàm đánh giá. Hiện tại chỉ hỗ trợ "f1".

    Returns:
        GridSearchResult chứa best_weights và toàn bộ kết quả.
    """
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
    tol = weight_step / 2

    best_metric_value = -1.0
    best_weights: dict[str, float] = {}
    all_results: list[tuple[dict[str, float], float]] = []

    combo_count = 0
    valid_count = 0

    for combo in product(weight_values, repeat=n_signals):
        combo_count += 1
        total = sum(combo)
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
