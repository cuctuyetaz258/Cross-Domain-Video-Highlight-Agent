"""Parse caption JSON3, fallback Whisper và lưu transcript"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from highlight_agent.schemas import (
    Chapter,
    TranscriptDocument,
    TranscriptSegment,
    TranscriptWord,
)

from .errors import MediaProcessingError


def save_transcript(document: TranscriptDocument, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(document.model_dump_json(indent=2), encoding="utf-8")
    temporary_path.replace(path)
    return path


def parse_youtube_json3(
    caption_path: str | Path,
    *,
    video_id: str,
    duration: float,
    chapters: list[Chapter] | None = None,
    language: str = "en",
) -> TranscriptDocument:
    try:
        payload = json.loads(Path(caption_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MediaProcessingError(f"could not parse YouTube caption file: {caption_path}") from exc

    segments: list[TranscriptSegment] = []
    for event in payload.get("events", []):
        fragments = event.get("segs") or []
        text = "".join(fragment.get("utf8", "") for fragment in fragments).replace("\n", " ").strip()
        if not text:
            continue

        start = max(0.0, float(event.get("tStartMs", 0)) / 1000)
        event_duration = max(0.01, float(event.get("dDurationMs", 0)) / 1000)
        end = min(duration, start + event_duration)
        if end <= start:
            continue

        words: list[TranscriptWord] = []
        timed_fragments = [fragment for fragment in fragments if fragment.get("utf8", "").strip()]
        if timed_fragments and all("tOffsetMs" in fragment for fragment in timed_fragments):
            for index, fragment in enumerate(timed_fragments):
                word_start = start + float(fragment["tOffsetMs"]) / 1000
                if index + 1 < len(timed_fragments):
                    word_end = start + float(timed_fragments[index + 1]["tOffsetMs"]) / 1000
                else:
                    word_end = end
                word_start = min(max(word_start, start), end)
                word_end = min(max(word_end, word_start), end)
                if word_end > word_start:
                    words.append(
                        TranscriptWord(
                            start=round(word_start, 3),
                            end=round(word_end, 3),
                            text=fragment["utf8"].strip(),
                        )
                    )

        segments.append(
            TranscriptSegment(
                id=len(segments),
                start=round(start, 3),
                end=round(end, 3),
                text=text,
                words=words,
            )
        )

    if not segments:
        raise MediaProcessingError("YouTube caption file did not contain usable transcript events")

    return TranscriptDocument(
        video_id=video_id,
        language=language,
        source="youtube_caption",
        duration=duration,
        segments=segments,
        chapters=chapters or [],
    )


def transcribe_with_whisper(
    audio_path: str | Path,
    *,
    video_id: str,
    duration: float,
    chapters: list[Chapter] | None = None,
    model_size: str = "base.en",
    model_factory: Callable[..., Any] | None = None,
) -> TranscriptDocument:
    if model_factory is None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise MediaProcessingError("faster-whisper is not installed") from exc
        model_factory = WhisperModel

    try:
        # Thử nạp mô hình bằng GPU với chuẩn nén int8_float16 để tiết kiệm VRAM
        model = model_factory(model_size, device="cuda", compute_type="int8_float16")
    except Exception as exc:
        # Nếu GPU hết VRAM hoặc lỗi CUDA, tự động chuyển về CPU
        print(f"[Whisper] GPU allocation failed ({exc}), falling back to CPU...")
        model = model_factory(model_size, device="cpu", compute_type="int8")
    raw_segments, info = model.transcribe(
        str(audio_path),
        beam_size=5,
        language="en",
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        word_timestamps=True,
        condition_on_previous_text=False,
    )

    segments: list[TranscriptSegment] = []
    for raw_segment in raw_segments:
        text = raw_segment.text.strip()
        if not text or raw_segment.end <= raw_segment.start:
            continue
        words = []
        for raw_word in raw_segment.words or []:
            word_text = raw_word.word.strip()
            if not word_text or raw_word.start is None or raw_word.end is None or raw_word.end <= raw_word.start:
                continue
            words.append(
                TranscriptWord(
                    start=round(raw_word.start, 3),
                    end=round(raw_word.end, 3),
                    text=word_text,
                )
            )
        
        words.sort(key=lambda w: w.start)
        
        segments.append(
            TranscriptSegment(
                id=len(segments),
                start=round(raw_segment.start, 3),
                end=round(raw_segment.end, 3),
                text=text,
                words=words,
            )
        )

    if not segments:
        raise MediaProcessingError("Whisper did not detect usable English speech")

    # Fix Pydantic validation error: faster-whisper sometimes outputs slightly out-of-order segments
    segments.sort(key=lambda s: s.start)
    
    # Re-assign sequential IDs after sorting
    for i, s in enumerate(segments):
        s.id = i

    detected_language = getattr(info, "language", None) or "en"
    return TranscriptDocument(
        video_id=video_id,
        language=detected_language,
        source="whisper",
        duration=duration,
        segments=segments,
        chapters=chapters or [],
    )
