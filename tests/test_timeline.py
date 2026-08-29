from pathlib import Path

from highlight_agent.features import build_feature_timeline, save_feature_timeline
from highlight_agent.schemas import AcousticFeatures, FeatureWindow, SemanticFeatures, VisualFeatures


def _acoustic(duration: float) -> AcousticFeatures:
    return AcousticFeatures(
        duration=duration,
        rms_mean=0.1,
        rms_peak=0.2,
        rms_p95=0.18,
        rms_std=0.02,
        voiced_ratio=0.8,
        silence_duration=0,
        silence_ratio=0,
    )


def test_timeline_combines_windows_and_writes_json(tmp_path: Path) -> None:
    timeline = build_feature_timeline(
        video_id="abcdefghijk",
        domain="lecture",
        duration=35.0,
        window_seconds=30.0,
        hop_seconds=30.0,
        acoustic=_acoustic(35.0),
        acoustic_windows=[
            FeatureWindow(start=0.0, end=30.0, acoustic=_acoustic(30.0)),
            FeatureWindow(start=30.0, end=35.0, acoustic=_acoustic(5.0)),
        ],
        semantic_windows=[
            SemanticFeatures(
                topic_relevance=0.5,
                semantic_novelty=0.2,
                tfidf_density=0.1,
                cue_score=0,
                raw_score=0.3,
                text_coverage=0.5,
            ),
            SemanticFeatures(
                topic_relevance=0.4,
                semantic_novelty=0.1,
                tfidf_density=0.2,
                cue_score=0,
                raw_score=0.25,
                text_coverage=0.5,
            ),
        ],
        visual_windows=[
            VisualFeatures(motion_score=1.0, method="pixel_diff", frame_count=30),
            VisualFeatures(motion_score=0.8, method="pixel_diff", frame_count=5),
        ],
    )

    output_path = save_feature_timeline(timeline, tmp_path / "features" / "features.json")

    assert output_path.is_file()
    assert '"schema_version": "1.0"' in output_path.read_text(encoding="utf-8")
    assert timeline.windows[-1].end == 35.0
    assert timeline.windows[0].semantic is not None
    assert timeline.windows[0].visual is not None
