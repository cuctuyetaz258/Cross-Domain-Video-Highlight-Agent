"""Prepare SumMe media, transcripts, and a canonical benchmark manifest."""

from __future__ import annotations

import argparse
import json
import random
import sys
import wave
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from highlight_agent.media import (  # noqa: E402
    MediaProcessingError,
    extract_audio_16k_mono,
    probe_duration,
    save_transcript,
    transcribe_with_whisper,
)
from highlight_agent.models.train_offline import load_summe  # noqa: E402
from highlight_agent.paths import portable_relative_path  # noqa: E402
from highlight_agent.schemas import TranscriptDocument  # noqa: E402
from scripts.prepare_tvsum import _video_index  # noqa: E402


def _write_silent_audio(path: Path, duration: float, sample_rate: int = 16_000) -> None:
    """Create a small canonical WAV for benchmark videos without an audio stream."""

    path.parent.mkdir(parents=True, exist_ok=True)
    remaining = max(1, round(duration * sample_rate))
    silence_chunk = b"\x00\x00" * sample_rate
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        while remaining:
            frame_count = min(remaining, sample_rate)
            handle.writeframes(silence_chunk[: frame_count * 2])
            remaining -= frame_count


def _pad_wav_to_duration(path: Path, duration: float) -> float:
    """Pad a canonical PCM WAV with silence when its stream ends before the video."""

    with wave.open(str(path), "rb") as handle:
        parameters = handle.getparams()
        frames = handle.readframes(handle.getnframes())
    target_frames = max(1, round(duration * parameters.framerate))
    missing_frames = target_frames - parameters.nframes
    if missing_frames <= 0:
        return 0.0
    silence = b"\x00" * missing_frames * parameters.nchannels * parameters.sampwidth
    temporary = path.with_suffix(path.suffix + ".tmp")
    with wave.open(str(temporary), "wb") as handle:
        handle.setparams(parameters)
        handle.writeframes(frames)
        handle.writeframes(silence)
    temporary.replace(path)
    return missing_frames / parameters.framerate


def _probe_fps(video_path: Path) -> float:
    capture = cv2.VideoCapture(str(video_path))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS)) if capture.isOpened() else 0.0
    finally:
        capture.release()
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"cannot determine FPS for {video_path}")
    return fps


def assign_summe_splits(
    records: list[dict[str, Any]],
    *,
    train_count: int,
    val_count: int,
    test_count: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Assign deterministic video-level SumMe splits."""

    if min(train_count, val_count, test_count) < 0:
        raise ValueError("split counts must be non-negative")
    if train_count <= 0 or val_count <= 0:
        raise ValueError("train_count and val_count must be positive")
    if train_count + val_count + test_count != len(records):
        raise ValueError("SumMe split counts must equal the selected video count")
    ordered = sorted(records, key=lambda record: str(record["video_id"]))
    random.Random(seed).shuffle(ordered)
    val_ids = {str(record["video_id"]) for record in ordered[:val_count]}
    test_ids = {
        str(record["video_id"])
        for record in ordered[val_count : val_count + test_count]
    }
    return [
        {
            **record,
            "split": (
                "val"
                if str(record["video_id"]) in val_ids
                else "test"
                if str(record["video_id"]) in test_ids
                else "train"
            ),
            "split_seed": seed,
        }
        for record in ordered
    ]


def prepare_summe(
    *,
    annotations_dir: str | Path,
    video_dir: str | Path,
    processed_dir: str | Path,
    manifest_path: str | Path,
    project_root: str | Path,
    limit: int = 25,
    train_count: int = 15,
    val_count: int = 5,
    test_count: int = 5,
    seed: int = 42,
    whisper_model_size: str = "small.en",
    force: bool = False,
) -> dict[str, Any]:
    """Create derived media and a train/val/test SumMe manifest."""

    root = Path(project_root).resolve()
    media_by_id = _video_index(Path(video_dir).resolve())
    records = sorted(load_summe(annotations_dir), key=lambda record: str(record["video_id"]))
    random.Random(seed).shuffle(records)
    if limit <= 0 or len(records) < limit:
        raise ValueError(f"SumMe provides {len(records)} records, requested {limit}")
    if train_count + val_count + test_count != limit:
        raise ValueError("train_count + val_count + test_count must equal limit")

    processed_root = Path(processed_dir).resolve()
    manifest_records: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    for record in records:
        if len(manifest_records) == limit:
            break
        video_id = str(record["video_id"])
        video_path = media_by_id.get(video_id)
        if video_path is None:
            results.append({"video_id": video_id, "status": "skipped_missing_media"})
            continue
        derived_dir = processed_root / video_id
        audio_path = derived_dir / "audio.wav"
        transcript_path = derived_dir / "transcript.json"
        try:
            duration = probe_duration(video_path)
            fps = _probe_fps(video_path)
            audio_available = True
            if force or not audio_path.is_file():
                try:
                    extract_audio_16k_mono(video_path, audio_path)
                except MediaProcessingError as exc:
                    if "does not contain any stream" not in str(exc):
                        raise
                    audio_available = False
                    _write_silent_audio(audio_path, duration)
            elif audio_path.stat().st_size <= 44:
                audio_available = False
            audio_padded_seconds = _pad_wav_to_duration(audio_path, duration)
            transcript_available = True
            if force or not transcript_path.is_file():
                try:
                    transcript = transcribe_with_whisper(
                        audio_path,
                        video_id=video_id,
                        duration=duration,
                        model_size=whisper_model_size,
                    )
                except (ValueError, MediaProcessingError) as exc:
                    if "Whisper did not detect usable" not in str(exc):
                        raise
                    transcript_available = False
                    transcript = TranscriptDocument(
                        video_id=video_id,
                        language="und",
                        source="whisper",
                        duration=duration,
                        segments=[],
                    )
                save_transcript(transcript, transcript_path)
            else:
                transcript_payload = json.loads(transcript_path.read_text(encoding="utf-8"))
                transcript_available = bool(transcript_payload.get("segments"))
            manifest_records.append(
                {
                    **record,
                    "frame_scores": np.asarray(record["frame_scores"], dtype=np.float32).tolist(),
                    "video_path": portable_relative_path(video_path, root),
                    "audio_path": portable_relative_path(audio_path, root),
                    "transcript_path": portable_relative_path(transcript_path, root),
                    "duration": duration,
                    "fps": fps,
                    "annotation_path": portable_relative_path(
                        Path(annotations_dir).resolve() / f"{video_id}.mat", root
                    ),
                    "whisper_model_size": whisper_model_size,
                    "audio_available": audio_available,
                    "audio_padded_seconds": audio_padded_seconds,
                    "transcript_available": transcript_available,
                }
            )
            results.append(
                {
                    "video_id": video_id,
                    "status": (
                        "prepared"
                        if audio_available and transcript_available
                        else "prepared_missing_modalities"
                    ),
                    "duration": duration,
                }
            )
        except Exception as exc:  # noqa: BLE001
            results.append({"video_id": video_id, "status": "failed", "error": str(exc)})

    if len(manifest_records) < limit:
        return {
            "ready": False,
            "prepared_count": len(manifest_records),
            "failed_count": sum(result["status"] == "failed" for result in results),
            "results": results,
        }
    assigned = assign_summe_splits(
        manifest_records,
        train_count=train_count,
        val_count=val_count,
        test_count=test_count,
        seed=seed,
    )
    destination = Path(manifest_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in assigned),
        encoding="utf-8",
    )
    temporary.replace(destination)
    return {
        "ready": True,
        "manifest": str(destination),
        "prepared_count": len(assigned),
        "split_counts": {
            split: sum(record["split"] == split for record in assigned)
            for split in ("train", "val", "test")
        },
        "failed_count": sum(result["status"] == "failed" for result in results),
        "results": results,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-dir", required=True)
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--processed-dir", default="data/raw/summe/processed")
    parser.add_argument("--manifest", default="data/manifests/summe.jsonl")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--train-count", type=int, default=15)
    parser.add_argument("--val-count", type=int, default=5)
    parser.add_argument("--test-count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--whisper-model-size", default="small.en")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report = prepare_summe(
        annotations_dir=args.annotations_dir,
        video_dir=args.video_dir,
        processed_dir=args.processed_dir,
        manifest_path=args.manifest,
        project_root=args.project_root,
        limit=args.limit,
        train_count=args.train_count,
        val_count=args.val_count,
        test_count=args.test_count,
        seed=args.seed,
        whisper_model_size=args.whisper_model_size,
        force=args.force,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
