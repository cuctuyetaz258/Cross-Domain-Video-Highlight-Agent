import numpy as np

from highlight_agent.features.alignment import build_feature_matrix
from highlight_agent.schemas import AcousticFeatures, FeatureWindow, InteractionFeatures, SpeakerTurn


def _acoustic(duration: float, *, rms: float, pitch: float | None, silence: float) -> AcousticFeatures:
    return AcousticFeatures(
        duration=duration,
        rms_mean=rms,
        rms_peak=rms,
        rms_p95=rms,
        rms_std=0.0,
        pitch_mean_hz=pitch,
        pitch_median_hz=pitch,
        pitch_std_hz=0.0 if pitch is not None else None,
        pitch_min_hz=pitch,
        pitch_max_hz=pitch,
        voiced_ratio=0.8,
        silence_duration=silence * duration,
        silence_ratio=silence,
    )


def _windows() -> tuple[AcousticFeatures, list[FeatureWindow]]:
    full = _acoustic(60.0, rms=0.2, pitch=150.0, silence=0.1)
    return full, [
        FeatureWindow(start=0.0, end=30.0, acoustic=_acoustic(30.0, rms=0.1, pitch=None, silence=0.2)),
        FeatureWindow(start=30.0, end=60.0, acoustic=_acoustic(30.0, rms=0.3, pitch=200.0, silence=0.0)),
    ]


def test_build_feature_matrix_shape_range_and_dtype() -> None:
    acoustic, windows = _windows()
    interaction = InteractionFeatures(
        duration=60.0,
        speaker_count=2,
        turn_count=1,
        turn_rate_per_minute=1.0,
        speech_duration=50.0,
        speech_ratio=50.0 / 60.0,
        turns=[SpeakerTurn(start=10.0, end=25.0, speaker="A"), SpeakerTurn(start=32.0, end=52.0, speaker="B")],
    )

    matrix = build_feature_matrix(
        acoustic,
        windows,
        scene_times=[12.0, 45.0],
        gesture_sparse=np.linspace(0.0, 1.0, 120, dtype=np.float32),
        word_scores=[(5.0, 8.0, 0.7), (35.0, 38.0, 1.0)],
        interaction=interaction,
        duration=60.0,
    )

    assert matrix.shape == (7, 600)
    assert matrix.dtype == np.float32
    assert np.all(np.isfinite(matrix))
    assert np.all((0.0 <= matrix) & (matrix <= 1.0))


def test_build_feature_matrix_without_interaction_has_zero_turn_channel() -> None:
    acoustic, windows = _windows()

    matrix = build_feature_matrix(
        acoustic,
        windows,
        scene_times=[],
        gesture_sparse=np.array([], dtype=np.float32),
        word_scores=[],
        interaction=None,
        duration=60.0,
    )

    assert matrix.shape == (7, 600)
    assert np.all(matrix[6] == 0.0)
