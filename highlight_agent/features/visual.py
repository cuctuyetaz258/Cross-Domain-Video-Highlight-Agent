"""
Tầng B — Trích xuất đặc trưng hình ảnh (Visual Feature Extraction)

Hai phương pháp:
  1. pixel_diff  : Tính trung bình |frame_t - frame_{t-1}| (đơn giản, nhanh)
  2. raft        : RAFT-Small deep learning optical flow (chính xác nhất, dùng GPU)

Cả hai đều trả về list[WindowVisualScore] — mỗi phần tử là điểm raw
cho một cửa sổ thời gian. Điểm raw chưa chuẩn hóa, cần qua
scoring.normalize_features() trước khi cộng tổng.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Data class kết quả
# ──────────────────────────────────────────────

@dataclass
class WindowVisualScore:
    """Điểm visual raw cho một cửa sổ thời gian."""

    start: float          # giây
    end: float            # giây
    motion_score: float   # raw score (chưa normalize) — đơn vị tuỳ method
    method: str           # "pixel_diff" | "raft"
    frame_count: int      # số frame thực sự được phân tích trong cửa sổ
    extra: dict = field(default_factory=dict)  # thông tin phụ (mean_magnitude, ...)

    @property
    def duration(self) -> float:
        return self.end - self.start


# ──────────────────────────────────────────────
# Hàm tiện ích nội bộ
# ──────────────────────────────────────────────

def _open_video(video_path: str | Path) -> cv2.VideoCapture:
    """Mở file video, raise nếu thất bại."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Không mở được video: {video_path}")
    return cap


def _read_gray_frame(cap: cv2.VideoCapture) -> np.ndarray | None:
    """Đọc 1 frame và chuyển sang grayscale. Trả về None nếu hết video."""
    ret, frame = cap.read()
    if not ret or frame is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _seek_to(cap: cv2.VideoCapture, second: float, fps: float) -> bool:
    """Nhảy đến giây `second` trong video (0-indexed frame)."""
    target_frame = int(second * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    return True


# ──────────────────────────────────────────────
# Phương pháp 1: Pixel Difference
# ──────────────────────────────────────────────

def _pixel_diff_score(
    cap: cv2.VideoCapture,
    fps: float,
    start: float,
    end: float,
    sample_fps: float,
) -> tuple[float, int]:
    """
    Tính trung bình sai số pixel tuyệt đối giữa các frame liên tiếp.

    Returns:
        (mean_diff, frame_count)  — mean_diff ∈ [0, 255]
    """
    _seek_to(cap, start, fps)

    step_sec = 1.0 / sample_fps
    timestamps = np.arange(start, end, step_sec)

    diffs: list[float] = []
    prev_gray: np.ndarray | None = None

    for t in timestamps:
        _seek_to(cap, t, fps)
        gray = _read_gray_frame(cap)
        if gray is None:
            break
        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray)
            diffs.append(float(diff.mean()))
        prev_gray = gray

    if not diffs:
        return 0.0, 0

    return float(np.mean(diffs)), len(diffs)


# ──────────────────────────────────────────────
# RAFT model lazy loader (chỉ import torch khi cần)
# ──────────────────────────────────────────────

class _RAFTModelLoader:
    """Singleton lazy loader cho RAFT-Small model.

    Chỉ import torch/torchvision và khởi tạo model khi .get_model()
    được gọi lần đầu. Các lần gọi sau tái sử dụng model đã load.
    """

    _model = None
    _transforms = None
    _device = None

    @classmethod
    def get_model(cls):
        """Trả về (model, transforms, device). Tự detect CUDA."""
        if cls._model is not None:
            return cls._model, cls._transforms, cls._device

        try:
            import torch
            from torchvision.models.optical_flow import (
                raft_small,
                Raft_Small_Weights,
            )
        except ImportError as exc:
            raise ImportError(
                "Cần cài 'torch' và 'torchvision' để dùng method='raft'. "
                "Chạy: pip install torch torchvision"
            ) from exc

        cls._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Workaround cho lỗi `cudnnGetLibConfig` trên Windows với PyTorch 2.x
        if cls._device.type == "cuda":
            torch.backends.cudnn.enabled = False

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
                    img1_t.unsqueeze(0), size=(new_h, new_w), mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
                img2_t = torch.nn.functional.interpolate(
                    img2_t.unsqueeze(0), size=(new_h, new_w), mode="bilinear",
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
            flow_mag = torch.sqrt(
                final_flow[0] ** 2 + final_flow[1] ** 2
            )
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
        duration = total_frames / fps if fps > 0 else 0.0

        if duration < window_size:
            logger.warning(
                "Video duration (%.1fs) nhỏ hơn window_size (%.1fs). "
                "Chỉ có 1 cửa sổ.",
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
                raise ValueError(
                    f"method không hợp lệ: {method!r}. "
                    "Chọn 'pixel_diff' hoặc 'raft'."
                )

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
