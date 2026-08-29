import numpy as np

from highlight_agent.features.overlap_blender import blend_scores


def test_blend_shape() -> None:
    timeline = blend_scores(np.ones(8, dtype=np.float32), T=120)

    assert timeline.shape == (120,)
    assert timeline.dtype == np.float32


def test_blend_max_overlap() -> None:
    scores = np.ones(10, dtype=np.float32)
    timeline = blend_scores(scores, T=140)

    assert timeline[40] == 1.0
    assert np.all(timeline[40:50] == 1.0)


def test_blend_no_nan() -> None:
    timeline = blend_scores(np.array([0.2, 0.8], dtype=np.float32), T=100)

    assert np.isfinite(timeline).all()
    assert np.all(timeline[60:] == 0.0)
