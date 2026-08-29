from highlight_agent.schemas import (
    AcousticFeatures,
    FeatureTimeline,
    FeatureWindow,
    InteractionFeatures,
    SpeakerTurn,
)
from scripts.build_diarization_review import build_review_html


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


def test_review_page_embeds_timeline_and_relative_video_source() -> None:
    interaction = InteractionFeatures(
        duration=30.0,
        speaker_count=3,
        turn_count=2,
        turn_rate_per_minute=4.0,
        speech_duration=30.0,
        speech_ratio=1.0,
        turns=[
            SpeakerTurn(start=0.0, end=10.0, speaker="SPEAKER_00"),
            SpeakerTurn(start=10.0, end=20.0, speaker="SPEAKER_01"),
            SpeakerTurn(start=20.0, end=30.0, speaker="SPEAKER_02"),
        ],
    )
    timeline = FeatureTimeline(
        video_id="abcdefghijk",
        domain="podcast",
        duration=30.0,
        window_seconds=30.0,
        hop_seconds=30.0,
        acoustic=_acoustic(30.0),
        interaction=interaction,
        windows=[FeatureWindow(start=0.0, end=30.0, acoustic=_acoustic(30.0), interaction=interaction)],
    )

    page = build_review_html(timeline, video_source="../source_video.mp4")

    assert 'src="../source_video.mp4"' in page
    assert '<video id="video"' in page
    assert "SPEAKER_00" in page
    assert "SPEAKER_01" in page
    assert "SPEAKER_02" in page
    assert '"SPEAKER_02": "speaker-2"' in page
    assert ".speaker-2 { background: hsl(" in page
    assert "0 lượt đổi speaker" not in page
    assert '"turn_count": 2' in page
