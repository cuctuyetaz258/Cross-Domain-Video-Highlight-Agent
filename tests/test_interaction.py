from types import SimpleNamespace

from highlight_agent.features.interaction import (
    extract_interaction_features,
    interaction_features_from_turns,
    select_device,
    windowed_interaction_features,
)
from highlight_agent.schemas import SpeakerTurn


def test_turn_taking_filters_short_turns_and_merges_same_speaker() -> None:
    features = interaction_features_from_turns(
        [
            SpeakerTurn(start=0.0, end=1.0, speaker="A"),
            SpeakerTurn(start=1.2, end=2.0, speaker="A"),
            SpeakerTurn(start=2.1, end=2.2, speaker="B"),
            SpeakerTurn(start=2.3, end=3.2, speaker="B"),
            SpeakerTurn(start=3.3, end=4.0, speaker="A"),
        ],
        duration=10.0,
    )

    assert [(turn.start, turn.end, turn.speaker) for turn in features.turns] == [
        (0.0, 2.0, "A"),
        (2.3, 3.2, "B"),
        (3.3, 4.0, "A"),
    ]
    assert features.speaker_count == 2
    assert features.turn_count == 2
    assert features.turn_rate_per_minute == 12.0


def test_select_device_prefers_cuda_then_mps_then_cpu() -> None:
    def torch_module(cuda: bool, mps: bool):
        return SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: cuda),
            backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
        )

    assert select_device(torch_module(cuda=True, mps=True)) == "cuda"
    assert select_device(torch_module(cuda=False, mps=True)) == "mps"
    assert select_device(torch_module(cuda=False, mps=False)) == "cpu"


def test_windowed_interaction_clips_turns_and_recomputes_turn_taking() -> None:
    features = interaction_features_from_turns(
        [
            SpeakerTurn(start=20.0, end=35.0, speaker="A"),
            SpeakerTurn(start=36.0, end=50.0, speaker="B"),
        ],
        duration=60.0,
    )

    windows = windowed_interaction_features(features)

    assert len(windows) == 2
    assert [(turn.start, turn.end, turn.speaker) for turn in windows[0].turns] == [(20.0, 30.0, "A")]
    assert [(turn.start, turn.end, turn.speaker) for turn in windows[1].turns] == [
        (0.0, 5.0, "A"),
        (6.0, 20.0, "B"),
    ]
    assert windows[0].turn_count == 0
    assert windows[1].turn_count == 1


def test_pyannote_wrapper_uses_exclusive_turns_and_selected_device() -> None:
    captured: dict[str, object] = {}

    class FakeAnnotation:
        def itertracks(self, yield_label: bool):
            assert yield_label is True
            yield SimpleNamespace(start=0.0, end=1.0), None, "SPEAKER_00"
            yield SimpleNamespace(start=1.1, end=2.0), None, "SPEAKER_01"

    class FakePipeline:
        def to(self, device: str):
            captured["device"] = device
            return self

        def __call__(self, waveform_input, **kwargs):
            captured["waveform_input"] = waveform_input
            captured["speaker_options"] = kwargs
            return SimpleNamespace(exclusive_speaker_diarization=FakeAnnotation())

    def pipeline_factory(model_id: str, *, token: str) -> FakePipeline:
        captured["model_id"] = model_id
        captured["token"] = token
        return FakePipeline()

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
        device=lambda device: device,
    )
    features = extract_interaction_features(
        "podcast.wav",
        hf_token="test-token",
        duration=10,
        num_speakers=2,
        pipeline_factory=pipeline_factory,
        torch_module=fake_torch,
        audio_loader=lambda path, _torch: {"waveform": "fixture-waveform", "sample_rate": 16000},
    )

    assert captured["device"] == "cpu"
    assert captured["token"] == "test-token"
    assert captured["waveform_input"] == {"waveform": "fixture-waveform", "sample_rate": 16000}
    assert captured["speaker_options"] == {"num_speakers": 2}
    assert features.turn_count == 1
    assert features.speaker_count == 2


def test_pyannote_wrapper_forwards_speaker_range() -> None:
    captured: dict[str, object] = {}

    class FakeAnnotation:
        def itertracks(self, yield_label: bool):
            return iter(())

    class FakePipeline:
        def to(self, device: str):
            return self

        def __call__(self, waveform_input, **kwargs):
            captured["waveform_input"] = waveform_input
            captured["speaker_options"] = kwargs
            return SimpleNamespace(exclusive_speaker_diarization=FakeAnnotation())

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
        device=lambda device: device,
    )
    extract_interaction_features(
        "podcast.wav",
        hf_token="test-token",
        duration=10,
        min_speakers=1,
        max_speakers=3,
        pipeline_factory=lambda *_args, **_kwargs: FakePipeline(),
        torch_module=fake_torch,
        audio_loader=lambda _path, _torch: {"waveform": "fixture", "sample_rate": 16000},
    )

    assert captured["speaker_options"] == {"min_speakers": 1, "max_speakers": 3}
