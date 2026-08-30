"""Build a leakage-safe manifest from completed 2-second custom annotations."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402

from highlight_agent.paths import portable_relative_path  # noqa: E402


def load_completed_annotation(path: str | Path) -> dict[str, Any]:
    """Load one fully scored CSV without converting graded labels to clips."""

    source = Path(path)
    with source.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"annotation is empty: {source}")
    required = {"video_id", "start_sec", "end_sec", "importance", "domain"}
    missing = sorted(required.difference(rows[0]))
    if missing:
        raise ValueError(f"annotation is missing columns: {', '.join(missing)}")

    video_id = str(rows[0]["video_id"]).strip()
    domain = str(rows[0]["domain"]).strip()
    if not video_id or domain not in {"lecture", "podcast"}:
        raise ValueError(f"invalid video_id/domain in {source}")
    segments: list[list[float]] = []
    expected_start = 0.0
    for line_number, row in enumerate(rows, start=2):
        if str(row.get("video_id", "")).strip() != video_id:
            raise ValueError(f"line {line_number} changes video_id")
        if str(row.get("domain", "")).strip() != domain:
            raise ValueError(f"line {line_number} changes domain")
        start = float(row["start_sec"])
        end = float(row["end_sec"])
        raw_score = str(row.get("importance", "")).strip()
        if not raw_score:
            raise ValueError(f"line {line_number} has no importance score")
        score = int(raw_score)
        if score not in {1, 2, 3, 4, 5}:
            raise ValueError(f"line {line_number} importance must be 1..5")
        if abs(start - expected_start) > 0.15 or end <= start:
            raise ValueError(f"line {line_number} has a gap, overlap, or invalid interval")
        segments.append([start, end, float(score)])
        expected_start = end
    return {
        "video_id": video_id,
        "dataset": "custom_lecture_podcast",
        "source": "custom_scores",
        "domain": domain,
        "duration": expected_start,
        "importance_segments": segments,
        "annotation_path": source,
        "annotation_type": "importance_1_to_5",
        "label_interval_sec": 2.0,
    }


def assign_group_folds(
    records: list[dict[str, Any]],
    *,
    fold: int,
    folds: int = 5,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Assign 3/1/1 train/val/test per domain for five-video domains."""

    if folds < 3 or not 0 <= fold < folds:
        raise ValueError("folds must be >= 3 and fold must be within range")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["domain"])].append(record)
    missing_domains = sorted({"lecture", "podcast"}.difference(grouped))
    if missing_domains:
        raise ValueError(
            "completed custom data is missing domains: " + ", ".join(missing_domains)
        )
    assigned: list[dict[str, Any]] = []
    for domain, domain_records in sorted(grouped.items()):
        if len(domain_records) < folds:
            raise ValueError(f"domain {domain!r} needs at least {folds} completed videos")
        ordered = sorted(domain_records, key=lambda item: str(item["video_id"]))
        random.Random(f"{seed}:{domain}").shuffle(ordered)
        test_id = str(ordered[fold % len(ordered)]["video_id"])
        val_id = str(ordered[(fold + 1) % len(ordered)]["video_id"])
        for record in ordered:
            video_id = str(record["video_id"])
            split = "test" if video_id == test_id else "val" if video_id == val_id else "train"
            assigned.append({**record, "split": split, "fold": fold, "split_seed": seed})
    return assigned


def _probe_media(video_path: Path) -> tuple[float, float]:
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise ValueError(f"cannot open video: {video_path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(fps) or fps <= 0 or not math.isfinite(frame_count) or frame_count <= 0:
            raise ValueError(f"video has invalid duration/FPS: {video_path}")
        return frame_count / fps, fps
    finally:
        capture.release()


def _align_annotation_to_media(record: dict[str, Any], duration: float) -> dict[str, Any]:
    """Use media duration as canonical and clip only annotation tail outside the file."""

    annotation_duration = float(record["duration"])
    aligned_segments = [
        [float(start), min(float(end), duration), float(score)]
        for start, end, score in record["importance_segments"]
        if float(start) < duration
    ]
    covered_until = float(aligned_segments[-1][1]) if aligned_segments else 0.0
    unlabeled_tail_seconds = max(0.0, duration - covered_until)
    if unlabeled_tail_seconds > 1e-6:
        aligned_segments.append([covered_until, duration, 3.0])
    return {
        **record,
        "duration": duration,
        "annotation_duration": annotation_duration,
        "importance_segments": aligned_segments,
        "unlabeled_tail_policy": "ignored_score_3",
        "unlabeled_tail_seconds": unlabeled_tail_seconds,
    }


def prepare_custom_manifest(
    *,
    annotations_dir: str | Path,
    media_root: str | Path,
    manifest_path: str | Path,
    project_root: str | Path,
    fold: int,
    folds: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    """Write one custom fold manifest when all required artifacts are ready."""

    root = Path(project_root).resolve()
    media = Path(media_root).resolve()
    completed: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for annotation in sorted(Path(annotations_dir).glob("*.csv")):
        try:
            record = load_completed_annotation(annotation)
        except Exception as exc:  # noqa: BLE001
            skipped.append({"file": annotation.name, "reason": str(exc)})
            continue
        workspace = media / str(record["video_id"])
        video_path = workspace / "source_video.mp4"
        audio_path = workspace / "audio.wav"
        transcript_path = workspace / "transcript.json"
        missing = [path.name for path in (video_path, audio_path, transcript_path) if not path.is_file()]
        if missing:
            skipped.append(
                {"file": annotation.name, "reason": f"missing media artifacts: {', '.join(missing)}"}
            )
            continue
        duration, fps = _probe_media(video_path)
        aligned_record = _align_annotation_to_media(record, duration)
        completed.append(
            {
                **aligned_record,
                "annotation_path": portable_relative_path(annotation.resolve(), root),
                "video_path": portable_relative_path(video_path, root),
                "audio_path": portable_relative_path(audio_path, root),
                "transcript_path": portable_relative_path(transcript_path, root),
                "fps": fps,
            }
        )

    try:
        assigned = assign_group_folds(completed, fold=fold, folds=folds, seed=seed)
    except ValueError as exc:
        return {
            "ready": False,
            "completed_video_count": len(completed),
            "skipped": skipped,
            "error": str(exc),
        }
    destination = Path(manifest_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    serializable = [
        {key: value for key, value in record.items() if not isinstance(value, Path)}
        for record in assigned
    ]
    temporary.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in serializable),
        encoding="utf-8",
    )
    temporary.replace(destination)
    split_counts = {
        split: sum(record["split"] == split for record in assigned)
        for split in ("train", "val", "test")
    }
    return {
        "ready": True,
        "manifest": str(destination),
        "completed_video_count": len(assigned),
        "split_counts": split_counts,
        "skipped": skipped,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-dir", default="data/annotations/raw")
    parser.add_argument("--media-root", default="output")
    parser.add_argument("--manifest", default="data/manifests/custom_fold0.jsonl")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report = prepare_custom_manifest(
        annotations_dir=args.annotations_dir,
        media_root=args.media_root,
        manifest_path=args.manifest,
        project_root=args.project_root,
        fold=args.fold,
        folds=args.folds,
        seed=args.seed,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
