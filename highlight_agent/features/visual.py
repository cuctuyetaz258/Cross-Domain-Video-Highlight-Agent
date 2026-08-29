"""
Tầng B — Trích xuất đặc trưng Visual Motion (Optical Flow & Pixel Difference)

Hỗ trợ 2 phương pháp:
  1. "pixel_diff": Nhanh, nhẹ CPU, tính chênh lệch pixel giữa các frame lấy mẫu.
  2. "raft": RAFT-Small (Deep Learning Optical Flow từ torchvision), chính xác nhất.

Output:
  List[WindowVisualScore] — mỗi window là 1 khoảng thời gian (mặc định 30s)
  kèm theo điểm motion thô (chưa normalize).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

import cv2
import numpy as np

logger = logging.getLogger(__name__)

VisualMethod = Literal["pixel_diff", "raft"]


# ──────────────────────────────────────────────
# Data class kết quả visual theo từng cửa sổ
# ──────────────────────────────────────────────


@dataclass
class WindowVisualScore:
    """Điểm chuyển động thị giác cho một cửa sổ thời gian."""

    start: float  # Giây bắt đầu
    end: float  # Giây kết thúc
    motion_score: float  # Điểm chuyển động thô (chưa chuẩn hóa)
    method: VisualMethod  # "pixel_diff" hoặc "raft"
    frame_count: int = 0  # Số frame đã phân tích trong window
    extra: dict = field(default_factory=dict)  # Metadata phụ (max, std, device, ...)

    @property
    def duration(self) -> float:
        return self.end - self.start

    def summary(self) -> str:
        return (
            f"[{self.start:.1f}s–{self.end:.1f}s] "
            f"motion={self.motion_score:.4f} "
            f"({self.method}, {self.frame_count} frames)"
        )


# ──────────────────────────────────────────────
# Helper: Đọc video an toàn
# ──────────────────────────────────────────────


def _open_video(video_path: str | Path) -> cv2.VideoCapture:
    """Mở video bằng OpenCV, raise ValueError nếu không mở được."""
    path_str = str(video_path)
    cap = cv2.VideoCapture(path_str)
    if not cap.isOpened():
        raise ValueError(f"Không thể mở video tại đường dẫn: {path_str}")
    return cap


def _seek_to(cap: cv2.VideoCapture, time_sec: float, fps: float) -> None:
    """Seek video đến vị trí time_sec (giây)."""
    frame_idx = int(time_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)


# ──────────────────────────────────────────────
# Phương pháp 1: Pixel Difference (CPU / Nhanh)
# ──────────────────────────────────────────────


def _pixel_diff_score(
    cap: cv2.VideoCapture,
    fps: float,
    start: float,
    end: float,
    sample_fps: float,
) -> tuple[float, int]:
    """
    Tính trung bình absolute difference giữa các frame liên tiếp trong window.

    Returns:
        (mean_diff_score, frame_count)
    """
    _seek_to(cap, start, fps)
    step_sec = 1.0 / sample_fps
    timestamps = np.arange(start, end, step_sec)

    diffs: list[float] = []
    prev_gray: np.ndarray | None = None

    for t in timestamps:
        _seek_to(cap, t, fps)
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        # Chuyển sang grayscale và resize nhỏ để tăng tốc tính toán
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (320, 180), interpolation=cv2.INTER_AREA)

        if prev_gray is not None:
            # Mean absolute pixel difference ∈ [0, 255]
            diff = float(np.mean(cv2.absdiff(gray, prev_gray)))
            diffs.append(diff)

        prev_gray = gray

    if not diffs:
        return 0.0, 0

    return float(np.mean(diffs)), len(diffs)


# ──────────────────────────────────────────────
# Lazy loader cho RAFT-Small (singleton để tiết kiệm VRAM)
# ──────────────────────────────────────────────


class _RAFTModelLoader:
    """Singleton quản lý model RAFT-Small chỉ nạp một lần vào bộ nhớ."""

    _model = None
    _transforms = None
    _device = None

    @classmethod
    def get_model(cls):
        if cls._model is not None:
            return cls._model, cls._transforms, cls._device

        try:
            import torch
            from torchvision.models.optical_flow import (
                Raft_Small_Weights,
                raft_small,
            )
        except ImportError as exc:
            raise ImportError("Để dùng method='raft', vui lòng cài: pip install torch torchvision") from exc

        cls._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weights = Raft_Small_Weights.DEFAULT
        cls._transforms = weights.transforms()

        cls._model = raft_small(weights=weights, progress=False)
        cls._model = cls._model.to(cls._device).eval()

        logger.info(
            "RAFT-Small loaded trên device=%s (VRAM ~300-500MB)",
            cls._device,
        )
        return cls._model, cls._transforms, cls._device

    @classmethod
    def reset(cls):
        """Giải phóng model khỏi bộ nhớ (dùng cho testing hoặc cleanup)."""
        cls._model = None
        cls._transforms = None
        cls._device = None


# ──────────────────────────────────────────────
# Phương pháp 2: RAFT-Small (Deep Learning Optical Flow)
# ──────────────────────────────────────────────


def _raft_score(
    cap: cv2.VideoCapture,
    fps: float,
    start: float,
    end: float,
    sample_fps: float,
) -> tuple[float, int, dict]:
    """
    Tính trung bình magnitude optical flow bằng RAFT-Small.

    Flow:
        1. Đọc cặp frame BGR từ OpenCV.
        2. Chuyển BGR → RGB → PyTorch Tensor (N,C,H,W).
        3. Resize về bội số 8 (yêu cầu của RAFT).
        4. Normalize qua transforms đi kèm pretrained weights.
        5. Chạy model → lấy flow cuối cùng → tính magnitude.

    Returns:
        (mean_magnitude, frame_count, extra_dict)
    """
    import torch

    model, transforms, device = _RAFTModelLoader.get_model()

    _seek_to(cap, start, fps)
    step_sec = 1.0 / sample_fps
    timestamps = np.arange(start, end, step_sec)

    magnitudes: list[float] = []
    prev_frame: np.ndarray | None = None  # BGR frame gốc

    for t in timestamps:
        _seek_to(cap, t, fps)
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        if prev_frame is not None:
            # BGR → RGB → Tensor float32 [0, 255]
            img1_rgb = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2RGB)
            img2_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            img1_t = torch.from_numpy(img1_rgb).permute(2, 0, 1).float()  # (C,H,W)
            img2_t = torch.from_numpy(img2_rgb).permute(2, 0, 1).float()

            # Resize về bội số 8 (yêu cầu kiến trúc RAFT)
            _, h, w = img1_t.shape
            new_h = (h // 8) * 8
            new_w = (w // 8) * 8
            if new_h != h or new_w != w:
                img1_t = torch.nn.functional.interpolate(
                    img1_t.unsqueeze(0),
                    size=(new_h, new_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
                img2_t = torch.nn.functional.interpolate(
                    img2_t.unsqueeze(0),
                    size=(new_h, new_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)

            # Apply transforms của pretrained weights
            img1_t, img2_t = transforms(img1_t, img2_t)

            # Inference (batch=1)
            with torch.no_grad():
                list_of_flows = model(
                    img1_t.unsqueeze(0).to(device),
                    img2_t.unsqueeze(0).to(device),
                )

            # Lấy flow cuối cùng (iteration cuối = chính xác nhất)
            final_flow = list_of_flows[-1][0]  # (2, H, W)
            flow_mag = torch.sqrt(final_flow[0] ** 2 + final_flow[1] ** 2)
            magnitudes.append(float(flow_mag.mean().cpu()))

        prev_frame = frame

    if not magnitudes:
        return 0.0, 0, {}

    return (
        float(np.mean(magnitudes)),
        len(magnitudes),
        {
            "max_magnitude": float(np.max(magnitudes)),
            "std_magnitude": float(np.std(magnitudes)),
            "device": str(device),
        },
    )


# ──────────────────────────────────────────────
# API công khai
# ──────────────────────────────────────────────


def extract_visual_scores(
    video_path: str | Path,
    window_size: float = 30.0,
    step_size: float | None = None,
    sample_fps: float = 1.0,
    method: Literal["pixel_diff", "raft"] = "pixel_diff",
    on_window: Callable[[WindowVisualScore], None] | None = None,
    duration: float | None = None,
) -> list[WindowVisualScore]:
    """
    Trích xuất điểm visual motion cho từng cửa sổ trượt.

    Args:
        video_path  : Đường dẫn tới video (.mp4, .webm, ...).
        window_size : Độ dài mỗi cửa sổ (giây). Mặc định 30s.
        step_size   : Bước dịch cửa sổ (giây). Mặc định = window_size (không chồng lấp).
        sample_fps  : Số frame lấy mẫu mỗi giây trong cửa sổ.
                      sample_fps=1 → 1 frame/giây, đủ cho Lecture/Podcast.
                      sample_fps=2 → 2 frame/giây, chi tiết hơn nhưng chậm hơn.
        method      : "pixel_diff" (nhanh) | "raft" (chính xác nhất, cần torch+GPU).
        on_window   : Callback nhận WindowVisualScore mỗi khi 1 cửa sổ xử lý xong.
                      Dùng để emit tiến độ real-time. Mặc định None (không callback).
        duration    : Mốc kết thúc dùng để đồng bộ cửa sổ với audio.

    Returns:
        Danh sách WindowVisualScore, mỗi phần tử ứng với 1 cửa sổ thời gian.
        motion_score CHƯA được chuẩn hóa — cần qua normalize_features() sau.

    Example:
        >>> scores = extract_visual_scores("lecture.mp4", window_size=30, method="pixel_diff")
        >>> for s in scores:
        ...     print(f"{s.start:.0f}s–{s.end:.0f}s → motion={s.motion_score:.2f}")
    """
    video_path = Path(video_path)
    step = step_size if step_size is not None else window_size

    cap = _open_video(video_path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        video_duration = total_frames / fps if fps > 0 else 0.0
        if duration is not None and duration <= 0:
            raise ValueError("duration phải lớn hơn 0 nếu được cung cấp")
        duration = duration if duration is not None else video_duration

        if duration < window_size:
            logger.warning(
                "Video duration (%.1fs) nhỏ hơn window_size (%.1fs). Chỉ có 1 cửa sổ.",
                duration,
                window_size,
            )

        results: list[WindowVisualScore] = []

        start = 0.0
        while start < duration:
            end = min(start + window_size, duration)

            if method == "pixel_diff":
                score, n_frames = _pixel_diff_score(cap, fps, start, end, sample_fps)
                extra: dict = {}
            elif method == "raft":
                score, n_frames, extra = _raft_score(cap, fps, start, end, sample_fps)
            else:
                raise ValueError(f"method không hợp lệ: {method!r}. Chọn 'pixel_diff' hoặc 'raft'.")

            results.append(
                WindowVisualScore(
                    start=round(start, 3),
                    end=round(end, 3),
                    motion_score=round(score, 4),
                    method=method,
                    frame_count=n_frames,
                    extra=extra,
                )
            )

            if on_window:
                on_window(results[-1])

            logger.debug(
                "[Visual/%s] %.1fs–%.1fs → score=%.3f (frames=%d)",
                method,
                start,
                end,
                score,
                n_frames,
            )

            start += step

        logger.info(
            "Trích xuất visual xong: %d cửa sổ, method=%s, video=%s",
            len(results),
            method,
            video_path.name,
        )
        return results

    finally:
        cap.release()


def scores_to_array(scores: list[WindowVisualScore]) -> np.ndarray:
    """
    Chuyển list[WindowVisualScore] → numpy array 1-D (motion_score).

    Dùng để đưa vào normalize_features() trong scoring.py.
    """
    return np.array([s.motion_score for s in scores], dtype=np.float32)
