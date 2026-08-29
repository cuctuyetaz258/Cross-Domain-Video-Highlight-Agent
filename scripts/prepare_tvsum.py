"""Tạo manifest TVSum và derived media cho feature-cache LTR"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from highlight_agent.media import (  # noqa: E402
    extract_audio_16k_mono,
    probe_duration,
    save_transcript,
    transcribe_with_whisper,
)
from highlight_agent.models.train_offline import load_tvsum  # noqa: E402
from highlight_agent.paths import portable_relative_path  # noqa: E402

VIDEO_SUFFIXES = (".m4v", ".mkv", ".mov", ".mp4", ".webm")


def _video_index(video_dir: Path) -> dict[str, Path]:
    """Lập chỉ mục media TVSum theo stem và từ chối ID bị trùng"""
    indexed: dict[str, Path] = {}
    for path in sorted(video_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in VIDEO_SUFFIXES:
            continue
        if path.stem in indexed:
            raise ValueError(f"multiple TVSum videos share ID {path.stem!r}")
        indexed[path.stem] = path
    return indexed


def _select_tvsum_candidates(
    records: list[dict[str, Any]],
    *,
    limit: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Chọn video TVSum với thứ tự tái lập và phủ category tối đa"""
    if limit <= 0:
        raise ValueError("limit phai lon hon 0")
    if len(records) < limit:
        raise ValueError(f"TVSum only provides {len(records)} records, need {limit}")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["category"])].append(record)
    generator = random.Random(seed)
    for category_records in grouped.values():
        category_records.sort(key=lambda record: str(record["video_id"]))
        generator.shuffle(category_records)

    selected: list[dict[str, Any]] = []
    while len(selected) < limit:
        progressed = False
        for category in sorted(grouped):
            if grouped[category] and len(selected) < limit:
                selected.append(grouped[category].pop())
                progressed = True
        if not progressed:
            break

    return selected


def _assign_tvsum_splits(
    selected: list[dict[str, Any]],
    *,
    train_count: int,
    val_count: int,
) -> list[dict[str, Any]]:
    """Chia tập video đã chuẩn bị thành train va validation theo category"""
    if train_count <= 0 or val_count <= 0:
        raise ValueError("train_count va val_count phai lon hon 0")
    if train_count + val_count != len(selected):
        raise ValueError("train_count + val_count phai bang so video da chon")

    limit = len(selected)
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in selected:
        by_category[str(record["category"])].append(record)
    validation_ids: set[str] = set()
    for category in sorted(by_category):
        category_records = by_category[category]
        requested = round(len(category_records) * val_count / limit)
        validation_ids.update(str(record["video_id"]) for record in category_records[:requested])
    ordered_ids = [str(record["video_id"]) for record in selected]
    for video_id in ordered_ids:
        if len(validation_ids) >= val_count:
            break
        validation_ids.add(video_id)
    while len(validation_ids) > val_count:
        validation_ids.remove(next(video_id for video_id in reversed(ordered_ids) if video_id in validation_ids))

    prepared: list[dict[str, Any]] = []
    for record in selected:
        prepared.append({**record, "split": "val" if str(record["video_id"]) in validation_ids else "train"})
    if sum(record["split"] == "train" for record in prepared) != train_count:
        raise RuntimeError("TVSum split did not produce the requested train count")
    return prepared


def select_tvsum_records(
    records: list[dict[str, Any]],
    *,
    limit: int,
    train_count: int,
    val_count: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Chọn và chia TVSum theo video với thứ tự tái lập và phủ category tối đa"""
    selected = _select_tvsum_candidates(records, limit=limit, seed=seed)
    return _assign_tvsum_splits(selected, train_count=train_count, val_count=val_count)


def prepare_tvsum(
    *,
    annotations_path: str | Path,
    video_dir: str | Path,
    processed_dir: str | Path,
    manifest_path: str | Path,
    project_root: str | Path,
    limit: int = 20,
    train_count: int = 16,
    val_count: int = 4,
    seed: int = 42,
    whisper_model_size: str = "small.en",
    force: bool = False,
) -> dict[str, Any]:
    """Sinh audio, transcript va manifest TVSum cho cache LTR"""
    root = Path(project_root).resolve()
    media_by_id = _video_index(Path(video_dir).resolve())
    records = load_tvsum(annotations_path)
    candidates = _select_tvsum_candidates(records, limit=len(records), seed=seed)
    processed_root = Path(processed_dir).resolve()
    results: list[dict[str, Any]] = []
    manifest_records: list[dict[str, Any]] = []

    for record in candidates:
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
            if force or not audio_path.is_file():
                extract_audio_16k_mono(video_path, audio_path)
            if force or not transcript_path.is_file():
                transcript = transcribe_with_whisper(
                    audio_path,
                    video_id=video_id,
                    duration=duration,
                    model_size=whisper_model_size,
                )
                save_transcript(transcript, transcript_path)
            manifest_records.append(
                {
                    "video_id": video_id,
                    "dataset": "tvsum",
                    "category": record["category"],
                    "domain": "benchmark",
                    "source": "tvsum",
                    "video_path": portable_relative_path(video_path, root),
                    "audio_path": portable_relative_path(audio_path, root),
                    "transcript_path": portable_relative_path(transcript_path, root),
                    "duration": duration,
                    "fps": record["fps"],
                    "frame_scores": np.asarray(record["frame_scores"], dtype=np.float32).tolist(),
                    "annotation_path": portable_relative_path(annotations_path, root),
                    "whisper_model_size": whisper_model_size,
                }
            )
            results.append({"video_id": video_id, "status": "prepared", "duration": duration})
        except Exception as exc:
            message = str(exc)
            status = "skipped_no_transcript" if "Whisper did not detect usable" in message else "failed"
            results.append({"video_id": video_id, "status": status, "error": message})

    if len(manifest_records) < limit:
        return {
            "ready": False,
            "prepared_count": len(manifest_records),
            "failed_count": sum(result["status"] == "failed" for result in results),
            "results": results,
        }
    manifest_records = _assign_tvsum_splits(
        manifest_records,
        train_count=train_count,
        val_count=val_count,
    )
    destination = Path(manifest_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in manifest_records),
        encoding="utf-8",
    )
    temporary.replace(destination)
    return {
        "ready": True,
        "prepared_count": len(manifest_records),
        "failed_count": sum(result["status"] == "failed" for result in results),
        "results": results,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True, help="TVSum tvsum50.mat path")
    parser.add_argument("--video-dir", required=True, help="Directory containing TVSum raw videos")
    parser.add_argument("--processed-dir", default="data/raw/tvsum/processed")
    parser.add_argument("--manifest", default="data/manifests/tvsum_smoke.jsonl")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--train-count", type=int, default=16)
    parser.add_argument("--val-count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--whisper-model-size", default="small.en")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report = prepare_tvsum(
        annotations_path=args.annotations,
        video_dir=args.video_dir,
        processed_dir=args.processed_dir,
        manifest_path=args.manifest,
        project_root=args.project_root,
        limit=args.limit,
        train_count=args.train_count,
        val_count=args.val_count,
        seed=args.seed,
        whisper_model_size=args.whisper_model_size,
        force=args.force,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
