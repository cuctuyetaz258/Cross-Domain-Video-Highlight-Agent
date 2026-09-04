"""Adapt Kaggle benchmark media into a runnable TVSum+SumMe feature manifest.

The source manifest remains the authority for labels and splits.  This helper
only resolves the public Kaggle video files, creates derived audio/transcript
artifacts below a caller-owned directory, and writes a new manifest pointing to
those artifacts.  It never mutates the read-only Kaggle input Dataset.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import wave
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from highlight_agent.media import (  # noqa: E402
    MediaProcessingError,
    extract_audio_16k_mono,
    probe_duration,
    save_transcript,
)
from highlight_agent.media.transcript import transcribe_with_whisper  # noqa: E402
from highlight_agent.schemas import TranscriptDocument  # noqa: E402

VIDEO_SUFFIXES = {".mp4", ".m4v", ".mkv", ".mov", ".webm"}


def normalize_media_id(value: str) -> str:
    """Match source names such as ``Base jumping`` and ``Base_jumping``."""

    return re.sub(r"[^a-z0-9]+", "", value.lower())


def load_records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def index_media(media_root: Path) -> dict[str, Path]:
    """Index media by a conservative normalized stem and reject ambiguity."""

    indexed: dict[str, Path] = {}
    for path in sorted(media_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in VIDEO_SUFFIXES:
            continue
        key = normalize_media_id(path.stem)
        if key in indexed:
            raise ValueError(
                f"ambiguous public media stem {key!r}: {indexed[key]} and {path}; "
                "provide a directory containing only the intended benchmark videos"
            )
        indexed[key] = path.resolve()
    if not indexed:
        raise FileNotFoundError(f"no supported video files found below {media_root}")
    return indexed


def adapt_records(records: list[dict[str, Any]], media_root: Path, derived_root: Path) -> list[dict[str, Any]]:
    """Return records whose paths point to mounted media and writable artifacts."""

    media_by_id = index_media(media_root)
    adapted: list[dict[str, Any]] = []
    missing: list[str] = []
    for record in records:
        video_id = str(record["video_id"])
        media = media_by_id.get(normalize_media_id(video_id))
        if media is None:
            missing.append(video_id)
            continue
        destination = derived_root / str(record["source"]) / video_id
        adapted.append(
            {
                **record,
                "video_path": str(media),
                "audio_path": str(destination / "audio.wav"),
                "transcript_path": str(destination / "transcript.json"),
            }
        )
    if missing:
        raise FileNotFoundError(f"public Dataset is missing {len(missing)} benchmark videos: {', '.join(missing)}")
    return adapted


def write_silent_audio(path: Path, *, duration: float, sample_rate: int = 16_000) -> None:
    """Create a valid mono WAV when a benchmark clip has no audio stream."""

    if duration <= 0:
        raise ValueError("silent audio duration must be positive")
    path.parent.mkdir(parents=True, exist_ok=True)
    sample_count = max(1, int(round(duration * sample_rate)))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * sample_count)


def materialize_records(records: list[dict[str, Any]], *, whisper_model: str, force: bool) -> list[dict[str, Any]]:
    """Create audio and a transcript for every resolved record, fail as a batch."""

    results: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        video_id = str(record["video_id"])
        audio_path = Path(record["audio_path"])
        transcript_path = Path(record["transcript_path"])
        try:
            duration = probe_duration(record["video_path"])
            audio_available = True
            if force or not audio_path.is_file():
                try:
                    extract_audio_16k_mono(record["video_path"], audio_path)
                except MediaProcessingError as exc:
                    if "does not contain any stream" not in str(exc):
                        raise
                    write_silent_audio(audio_path, duration=duration)
                    audio_available = False
            if force or not transcript_path.is_file():
                if not audio_available:
                    transcript = TranscriptDocument(
                        video_id=video_id, language="und", source="no_audio", duration=duration, segments=[]
                    )
                    transcript_available = False
                    save_transcript(transcript, transcript_path)
                    record["audio_available"] = False
                    record["transcript_available"] = False
                    results.append({"video_id": video_id, "status": "ready", "audio_available": False, "transcript_available": False})
                    print(f"[{index}/{len(records)}] {json.dumps(results[-1], sort_keys=True)}", flush=True)
                    continue
                try:
                    transcript = transcribe_with_whisper(
                        audio_path, video_id=video_id, duration=duration, model_size=whisper_model
                    )
                    transcript_available = True
                except MediaProcessingError as exc:
                    # Silent/non-English clips still need a valid, zero-TF-IDF transcript.
                    if "did not detect usable" not in str(exc):
                        raise
                    transcript = TranscriptDocument(
                        video_id=video_id, language="und", source="whisper", duration=duration, segments=[]
                    )
                    transcript_available = False
                save_transcript(transcript, transcript_path)
            else:
                transcript_available = True
            record["audio_available"] = audio_available
            record["transcript_available"] = transcript_available
            results.append(
                {
                    "video_id": video_id,
                    "status": "ready",
                    "audio_available": audio_available,
                    "transcript_available": transcript_available,
                }
            )
        except Exception as exc:
            results.append({"video_id": video_id, "status": "failed", "error": str(exc)})
        print(f"[{index}/{len(records)}] {json.dumps(results[-1], sort_keys=True)}", flush=True)
    failures = [result for result in results if result["status"] == "failed"]
    if failures:
        raise RuntimeError(f"failed to materialize {len(failures)} benchmark records")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--media-root", required=True, help="Read-only root of the public Kaggle video Dataset.")
    parser.add_argument("--derived-root", required=True, help="Writable directory for audio and transcripts.")
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--whisper-model", default="small.en")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--prepare-media", action="store_true", help="Extract audio and transcribe after validating paths.")
    args = parser.parse_args()

    records = adapt_records(load_records(Path(args.source_manifest)), Path(args.media_root), Path(args.derived_root))
    output = Path(args.output_manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records), encoding="utf-8")
    print(f"adapted {len(records)} records to {output}", flush=True)
    if args.prepare_media:
        materialize_records(records, whisper_model=args.whisper_model, force=args.force)
        # Persist the observed audio/transcript availability for reproducibility.
        output.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records), encoding="utf-8")


if __name__ == "__main__":
    main()
