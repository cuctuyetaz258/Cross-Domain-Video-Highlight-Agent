"""Diarization Pyannote và feature đổi lượt nói cho audio podcast"""

import os
from collections.abc import Callable, Iterable
from itertools import pairwise
from pathlib import Path
from typing import Any

from highlight_agent.media.audio import probe_duration
from highlight_agent.schemas import InteractionFeatures, SpeakerTurn

DEFAULT_DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"
DEFAULT_MIN_TURN_DURATION = 0.30
DEFAULT_MAX_SAME_SPEAKER_GAP = 0.50


def select_device(torch_module: Any) -> str:
    """Ưu tiên CUDA, rồi Apple MPS, cuối cùng mới dùng CPU"""

    if torch_module.cuda.is_available():
        return "cuda"
    if torch_module.backends.mps.is_available():
        return "mps"
    return "cpu"


def normalize_speaker_turns(
    turns: Iterable[SpeakerTurn],
    *,
    duration: float,
    min_turn_duration: float = DEFAULT_MIN_TURN_DURATION,
    max_same_speaker_gap: float = DEFAULT_MAX_SAME_SPEAKER_GAP,
) -> list[SpeakerTurn]:
    """Lọc lượt nói quá ngắn và gộp các lượt liền kề của cùng speaker"""

    if duration <= 0:
        raise ValueError("duration must be positive")
    if min_turn_duration < 0 or max_same_speaker_gap < 0:
        raise ValueError("turn duration and speaker gap must be non-negative")

    valid = sorted(
        (
            turn
            for turn in turns
            if turn.end <= duration + 1e-6 and turn.end - turn.start >= min_turn_duration
        ),
        key=lambda turn: (turn.start, turn.end, turn.speaker),
    )
    normalized: list[SpeakerTurn] = []
    for turn in valid:
        if (
            normalized
            and turn.speaker == normalized[-1].speaker
            and turn.start - normalized[-1].end <= max_same_speaker_gap
        ):
            previous = normalized[-1]
            normalized[-1] = SpeakerTurn(
                start=previous.start,
                end=max(previous.end, turn.end),
                speaker=previous.speaker,
            )
        else:
            normalized.append(turn)
    return normalized


def interaction_features_from_turns(
    turns: Iterable[SpeakerTurn],
    *,
    duration: float,
    min_turn_duration: float = DEFAULT_MIN_TURN_DURATION,
    max_same_speaker_gap: float = DEFAULT_MAX_SAME_SPEAKER_GAP,
) -> InteractionFeatures:
    """Tính feature tương tác podcast từ các lượt nói đã diarization"""

    normalized = normalize_speaker_turns(
        turns,
        duration=duration,
        min_turn_duration=min_turn_duration,
        max_same_speaker_gap=max_same_speaker_gap,
    )
    turn_count = sum(
        current.speaker != previous.speaker
        for previous, current in pairwise(normalized)
    )
    speech_duration = sum(turn.end - turn.start for turn in normalized)
    return InteractionFeatures(
        duration=duration,
        speaker_count=len({turn.speaker for turn in normalized}),
        turn_count=turn_count,
        turn_rate_per_minute=turn_count / (duration / 60),
        speech_duration=speech_duration,
        speech_ratio=speech_duration / duration,
        turns=normalized,
    )


def windowed_interaction_features(
    features: InteractionFeatures,
    *,
    window_seconds: float = 30.0,
    hop_seconds: float = 30.0,
    min_turn_duration: float = DEFAULT_MIN_TURN_DURATION,
    max_same_speaker_gap: float = DEFAULT_MAX_SAME_SPEAKER_GAP,
) -> list[InteractionFeatures]:
    """Đưa các lượt nói diarization vào các cửa sổ riêng để chấm điểm sau này"""

    if window_seconds <= 0 or hop_seconds <= 0:
        raise ValueError("window_seconds and hop_seconds must be positive")

    windowed: list[InteractionFeatures] = []
    start = 0.0
    while start < features.duration:
        end = min(start + window_seconds, features.duration)
        turns = [
            SpeakerTurn(
                start=max(turn.start, start) - start,
                end=min(turn.end, end) - start,
                speaker=turn.speaker,
            )
            for turn in features.turns
            if turn.start < end and turn.end > start
        ]
        windowed.append(
            interaction_features_from_turns(
                turns,
                duration=end - start,
                min_turn_duration=min_turn_duration,
                max_same_speaker_gap=max_same_speaker_gap,
            )
        )
        start += hop_seconds
    return windowed


def _turns_from_annotation(annotation: Any) -> list[SpeakerTurn]:
    return [
        SpeakerTurn(start=float(segment.start), end=float(segment.end), speaker=str(speaker))
        for segment, _, speaker in annotation.itertracks(yield_label=True)
    ]


def extract_interaction_features(
    audio_path: str | Path,
    *,
    hf_token: str | None = None,
    model_id: str = DEFAULT_DIARIZATION_MODEL,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    min_turn_duration: float = DEFAULT_MIN_TURN_DURATION,
    max_same_speaker_gap: float = DEFAULT_MAX_SAME_SPEAKER_GAP,
    duration: float | None = None,
    pipeline_factory: Callable[..., Any] | None = None,
    torch_module: Any | None = None,
) -> InteractionFeatures:
    """Chạy Pyannote và tính lượt đổi speaker từ diarization độc quyền

    `pipeline_factory`, `torch_module` và `duration` có thể truyền vào để unit test
    Code chạy thật sử dụng các giá trị mặc định
    """

    token = hf_token or os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN is required for Pyannote speaker diarization")
    if num_speakers is not None and num_speakers < 1:
        raise ValueError("num_speakers must be positive")
    if min_speakers is not None and min_speakers < 1:
        raise ValueError("min_speakers must be positive")
    if max_speakers is not None and max_speakers < 1:
        raise ValueError("max_speakers must be positive")
    if min_speakers is not None and max_speakers is not None and min_speakers > max_speakers:
        raise ValueError("min_speakers cannot exceed max_speakers")
    if num_speakers is not None and (min_speakers is not None or max_speakers is not None):
        raise ValueError("num_speakers cannot be combined with min_speakers or max_speakers")

    if torch_module is None:
        import torch

        torch_module = torch
    if pipeline_factory is None:
        from pyannote.audio import Pipeline

        pipeline_factory = Pipeline.from_pretrained

    device_name = select_device(torch_module)
    pipeline = pipeline_factory(model_id, token=token)
    pipeline.to(torch_module.device(device_name))
    speaker_options = {
        name: value
        for name, value in {
            "num_speakers": num_speakers,
            "min_speakers": min_speakers,
            "max_speakers": max_speakers,
        }.items()
        if value is not None
    }
    diarization_output = pipeline(str(audio_path), **speaker_options)
    annotation = getattr(
        diarization_output,
        "exclusive_speaker_diarization",
        getattr(diarization_output, "speaker_diarization", diarization_output),
    )
    audio_duration = probe_duration(audio_path) if duration is None else duration
    return interaction_features_from_turns(
        _turns_from_annotation(annotation),
        duration=audio_duration,
        min_turn_duration=min_turn_duration,
        max_same_speaker_gap=max_same_speaker_gap,
    )
