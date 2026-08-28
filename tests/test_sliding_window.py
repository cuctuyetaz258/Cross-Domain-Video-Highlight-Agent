import numpy as np
import pytest

from highlight_agent.features.sliding_window import extract_windows

torch = pytest.importorskip("torch")


def test_unfold_k_count() -> None:
    feature_matrix = np.ones((7, 120), dtype=np.float32)

    windows = extract_windows(feature_matrix)

    assert windows.shape[0] == (120 - 50) // 10 + 1


def test_unfold_output_shape_and_means() -> None:
    feature_matrix = np.arange(7 * 120, dtype=np.float32).reshape(7, 120)

    windows = extract_windows(feature_matrix, device="cpu")

    assert windows.shape == (8, 7)
    assert windows.dtype == torch.float32
    assert torch.allclose(windows[0], torch.tensor([24.5, 144.5, 264.5, 384.5, 504.5, 624.5, 744.5]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA khong kha dung")
def test_unfold_device_gpu() -> None:
    windows = extract_windows(np.ones((7, 50), dtype=np.float32), device="cuda")

    assert windows.device.type == "cuda"
