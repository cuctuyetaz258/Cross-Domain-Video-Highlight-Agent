"""Tạo các vector đặc trưng cửa sổ cho LTR inference"""

from __future__ import annotations

import numpy as np
import torch


def extract_windows(
    feature_matrix: np.ndarray,
    window_size: int = 50,
    hop_size: int = 10,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """Mean-pool feature timeline 7 kenh thanh cac cua so co hop"""
    if feature_matrix.ndim != 2:
        raise ValueError("feature_matrix phai la mang 2 chieu")
    if window_size <= 0 or hop_size <= 0:
        raise ValueError("window_size va hop_size phai lon hon 0")

    channel_count, frame_count = feature_matrix.shape
    if channel_count != 7:
        raise ValueError("feature_matrix phai co dung 7 kenh")
    if frame_count < window_size:
        raise ValueError("feature_matrix ngan hon window_size")

    target_device = torch.device(device) if device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    timeline = torch.as_tensor(feature_matrix, dtype=torch.float32, device=target_device)
    windows = timeline.unfold(dimension=1, size=window_size, step=hop_size)
    return windows.permute(1, 0, 2).mean(dim=-1)
