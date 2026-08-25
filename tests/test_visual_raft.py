"""
Unit tests cho visual.py (Pixel Difference & RAFT Optical Flow).
Tự động skip RAFT tests nếu máy không có torch/torchvision.
Chạy bằng: pytest tests/test_visual_raft.py -v
"""

from pathlib import Path

import numpy as np
import pytest

from highlight_agent.features.visual import (
    WindowVisualScore,
    extract_visual_scores,
    scores_to_array,
)


# ──────────────────────────────────────────────
# Fixture tạo dummy video ngắn
# ──────────────────────────────────────────────

@pytest.fixture()
def dummy_video(tmp_path: Path) -> Path:
    """Tạo video ngắn 2 giây (10 fps) với các frame random."""
    import cv2

    video_path = tmp_path / "dummy.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 10, (128, 128))
    for _ in range(20):
        frame = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return video_path


# ──────────────────────────────────────────────
# Test Pixel Difference
# ──────────────────────────────────────────────

class TestPixelDiffVisualScore:
    def test_returns_results(self, dummy_video: Path):
        results = extract_visual_scores(
            dummy_video, window_size=2.0, sample_fps=2.0, method="pixel_diff"
        )
        assert len(results) >= 1
        assert isinstance(results[0], WindowVisualScore)
        assert results[0].method == "pixel_diff"
        assert results[0].motion_score >= 0.0

    def test_scores_to_array(self, dummy_video: Path):
        results = extract_visual_scores(
            dummy_video, window_size=2.0, sample_fps=2.0, method="pixel_diff"
        )
        arr = scores_to_array(results)
        assert isinstance(arr, np.ndarray)
        assert len(arr) == len(results)
        assert arr.dtype == np.float32

    def test_callback_invoked(self, dummy_video: Path):
        captured = []
        extract_visual_scores(
            dummy_video,
            window_size=2.0,
            sample_fps=2.0,
            method="pixel_diff",
            on_window=lambda score: captured.append(score),
        )
        assert len(captured) >= 1

    def test_invalid_method_raises(self, dummy_video: Path):
        with pytest.raises(ValueError, match="method không hợp lệ"):
            extract_visual_scores(
                dummy_video, window_size=2.0, method="invalid_method"  # type: ignore
            )


# ──────────────────────────────────────────────
# Test RAFT Model Loader & Inference
# ──────────────────────────────────────────────

class TestRAFTModelLoader:
    def test_model_loads_or_skips(self):
        pytest.importorskip("torch", reason="torch không có — skip RAFT tests")
        pytest.importorskip("torchvision", reason="torchvision không có — skip RAFT tests")

        from highlight_agent.features.visual import _RAFTModelLoader

        _RAFTModelLoader.reset()
        model, transforms, device = _RAFTModelLoader.get_model()
        assert model is not None
        assert transforms is not None
        assert str(device) in ("cpu", "cuda")

    def test_singleton_returns_same_model(self):
        pytest.importorskip("torch", reason="torch không có — skip RAFT tests")
        pytest.importorskip("torchvision", reason="torchvision không có — skip RAFT tests")

        from highlight_agent.features.visual import _RAFTModelLoader

        _RAFTModelLoader.reset()
        m1, _, _ = _RAFTModelLoader.get_model()
        m2, _, _ = _RAFTModelLoader.get_model()
        assert m1 is m2
        _RAFTModelLoader.reset()


class TestRAFTScoreWithDummyVideo:
    def test_raft_returns_results(self, dummy_video: Path):
        pytest.importorskip("torch", reason="torch không có — skip RAFT tests")
        pytest.importorskip("torchvision", reason="torchvision không có — skip RAFT tests")

        results = extract_visual_scores(
            dummy_video, window_size=2.0, sample_fps=2.0, method="raft"
        )
        assert len(results) >= 1
        assert results[0].method == "raft"
        assert results[0].motion_score >= 0.0
        assert "device" in results[0].extra
