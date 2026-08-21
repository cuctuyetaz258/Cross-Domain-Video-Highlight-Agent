"""
Unit tests cho RAFT integration trong visual.py.
Tự động skip nếu máy không có torch/torchvision.
Chạy bằng: pytest tests/test_visual_raft.py -v
"""

import pytest

torch = pytest.importorskip("torch", reason="torch không có — skip RAFT tests")
tv = pytest.importorskip("torchvision", reason="torchvision không có — skip RAFT tests")


# ──────────────────────────────────────────────
# Test _RAFTModelLoader
# ──────────────────────────────────────────────

class TestRAFTModelLoader:
    """Test lazy loader singleton."""

    def test_model_loads_successfully(self):
        """Model load lần đầu thành công."""
        from highlight_agent.features.visual import _RAFTModelLoader

        _RAFTModelLoader.reset()
        model, transforms, device = _RAFTModelLoader.get_model()
        assert model is not None
        assert transforms is not None
        assert str(device) in ("cpu", "cuda")

    def test_singleton_returns_same_model(self):
        """Gọi get_model() 2 lần phải trả về cùng instance."""
        from highlight_agent.features.visual import _RAFTModelLoader

        _RAFTModelLoader.reset()
        m1, _, _ = _RAFTModelLoader.get_model()
        m2, _, _ = _RAFTModelLoader.get_model()
        assert m1 is m2

    def test_reset_clears_model(self):
        """Sau reset(), get_model() phải load lại model mới."""
        from highlight_agent.features.visual import _RAFTModelLoader

        _RAFTModelLoader.reset()
        m1, _, _ = _RAFTModelLoader.get_model()
        _RAFTModelLoader.reset()
        assert _RAFTModelLoader._model is None


# ──────────────────────────────────────────────
# Test luồng end-to-end với dummy video
# ──────────────────────────────────────────────

class TestRAFTScoreWithDummyVideo:
    """Test luồng chạy end-to-end bằng video dummy (random frames)."""

    @pytest.fixture()
    def dummy_video(self, tmp_path):
        """Tạo video ngắn 2 giây (10 fps) với các frame random."""
        import cv2
        import numpy as np

        video_path = tmp_path / "dummy.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, 10, (128, 128))
        for _ in range(20):  # 20 frames = 2 giây
            frame = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
            writer.write(frame)
        writer.release()
        return video_path

    def test_raft_returns_results(self, dummy_video):
        """extract_visual_scores(method='raft') phải trả về ít nhất 1 window."""
        from highlight_agent.features.visual import extract_visual_scores

        results = extract_visual_scores(
            dummy_video, window_size=2.0, sample_fps=2.0, method="raft"
        )
        assert len(results) >= 1
        assert results[0].method == "raft"
        assert results[0].motion_score >= 0.0

    def test_raft_extra_has_device(self, dummy_video):
        """extra dict phải chứa key 'device'."""
        from highlight_agent.features.visual import extract_visual_scores

        results = extract_visual_scores(
            dummy_video, window_size=2.0, sample_fps=2.0, method="raft"
        )
        assert "device" in results[0].extra

    def test_raft_extra_has_magnitude_stats(self, dummy_video):
        """extra dict phải chứa max_magnitude và std_magnitude."""
        from highlight_agent.features.visual import extract_visual_scores

        results = extract_visual_scores(
            dummy_video, window_size=2.0, sample_fps=2.0, method="raft"
        )
        assert "max_magnitude" in results[0].extra
        assert "std_magnitude" in results[0].extra

    def test_invalid_method_raises(self, dummy_video):
        """method không hợp lệ phải raise ValueError."""
        from highlight_agent.features.visual import extract_visual_scores

        with pytest.raises(ValueError, match="method không hợp lệ"):
            extract_visual_scores(
                dummy_video, window_size=2.0, method="optical_flow"  # type: ignore
            )
