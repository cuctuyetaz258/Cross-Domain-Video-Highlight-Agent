"""Validate LTR training manifests and report split/domain/label statistics."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402

from highlight_agent.backend import load_transcript  # noqa: E402
from highlight_agent.models.train_offline import (  # noqa: E402
    compute_lref,
    create_window_labels,
    load_training_manifest,
)

VALID_DOMAINS = {"benchmark", "lecture", "podcast", "standup"}
VALID_SPLITS = {"train", "val", "test"}
PATH_FIELDS = ("video_path", "audio_path", "transcript_path")


def resolve_record_path(value: str | Path, project_root: str | Path) -> Path:
    """Resolve manifest paths relative to the explicitly selected project root."""

    path = Path(value)
    return path.resolve() if path.is_absolute() else (Path(project_root) / path).resolve()


def probe_video(path: str | Path) -> tuple[float, float]:
    """Return duration and FPS using OpenCV, failing on unreadable media."""

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError("video cannot be opened")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or frame_count <= 0:
            raise ValueError("video has invalid FPS or frame count")
        return frame_count / fps, fps
    finally:
        capture.release()


def _empty_bucket() -> dict[str, Any]:
    return {
        "videos": 0,
        "duration_seconds": 0.0,
        "positive_windows": 0,
        "negative_windows": 0,
        "ignored_windows": 0,
        "pairable_videos": 0,
        "pair_count": 0,
        "domains": Counter(),
        "sources": Counter(),
    }


def validate_manifest(
    manifest_path: str | Path,
    *,
    project_root: str | Path = ".",
    window_sec: float = 5.0,
    hop_sec: float = 1.0,
    minimum_videos: int = 1,
    required_domains: set[str] | None = None,
) -> dict[str, Any]:
    """Validate one manifest without mutating media or cache state."""

    manifest = Path(manifest_path).resolve()
    root = Path(project_root).resolve()
    errors: list[str] = []
    warnings: list[str] = []
    try:
        records = load_training_manifest(manifest)
    except Exception as exc:
        return {
            "valid": False,
            "manifest": str(manifest),
            "project_root": str(root),
            "errors": [f"cannot load manifest: {exc}"],
            "warnings": [],
            "video_count": 0,
            "splits": {},
        }

    if len(records) < minimum_videos:
        errors.append(f"manifest has {len(records)} videos; minimum is {minimum_videos}")

    seen_ids: dict[str, int] = {}
    seen_paths: dict[Path, tuple[str, str]] = {}
    stats: dict[str, dict[str, Any]] = defaultdict(_empty_bucket)
    valid_records: list[dict[str, Any]] = []

    for index, record in enumerate(records, start=1):
        prefix = f"line {index}"
        video_id = str(record.get("video_id", "")).strip()
        split = str(record.get("split", "")).strip()
        domain = str(record.get("domain", "")).strip()
        source = str(record.get("source", "")).strip()
        if not video_id:
            errors.append(f"{prefix}: video_id is empty")
            continue
        if video_id in seen_ids:
            errors.append(
                f"{prefix}: duplicate video_id {video_id!r} (first seen on line {seen_ids[video_id]})"
            )
        else:
            seen_ids[video_id] = index
        if split not in VALID_SPLITS:
            errors.append(f"{prefix} {video_id}: invalid split {split!r}")
        if domain not in VALID_DOMAINS:
            errors.append(f"{prefix} {video_id}: invalid domain {domain!r}")
        if not source:
            errors.append(f"{prefix} {video_id}: source is empty")
        if source == "tvsum" and not str(record.get("category", "")).strip():
            errors.append(f"{prefix} {video_id}: TVSum record is missing category")
        if source == "tvsum" and domain != "benchmark":
            errors.append(f"{prefix} {video_id}: TVSum domain must be 'benchmark'")

        resolved: dict[str, Path] = {}
        for field in PATH_FIELDS:
            value = record.get(field)
            if not value:
                errors.append(f"{prefix} {video_id}: missing {field}")
                continue
            path = resolve_record_path(value, root)
            resolved[field] = path
            if not path.is_file():
                errors.append(f"{prefix} {video_id}: {field} not found: {path}")

        video_path = resolved.get("video_path")
        duration = float(record.get("duration") or 0.0)
        fps = float(record.get("fps") or 0.0)
        if video_path and video_path.is_file():
            prior = seen_paths.get(video_path)
            if prior and prior != (video_id, split):
                errors.append(
                    f"{prefix} {video_id}: media path duplicates {prior[0]!r} across records/splits"
                )
            else:
                seen_paths[video_path] = (video_id, split)
            try:
                probed_duration, probed_fps = probe_video(video_path)
                if duration <= 0:
                    duration = probed_duration
                    record["duration"] = duration
                elif abs(duration - probed_duration) > max(2.0, probed_duration * 0.01):
                    errors.append(
                        f"{prefix} {video_id}: manifest duration {duration:.3f}s differs from media "
                        f"{probed_duration:.3f}s"
                    )
                if fps <= 0:
                    fps = probed_fps
                    record["fps"] = fps
                elif abs(fps - probed_fps) > 0.1:
                    errors.append(
                        f"{prefix} {video_id}: manifest FPS {fps:.3f} differs from media {probed_fps:.3f}"
                    )
            except Exception as exc:
                errors.append(f"{prefix} {video_id}: cannot probe video: {exc}")
        if duration <= 0:
            errors.append(f"{prefix} {video_id}: duration must be positive")

        transcript_path = resolved.get("transcript_path")
        if transcript_path and transcript_path.is_file():
            try:
                transcript = load_transcript(transcript_path)
                if transcript.video_id != video_id:
                    warnings.append(
                        f"{prefix} {video_id}: transcript video_id is {transcript.video_id!r}"
                    )
                if duration > 0 and abs(transcript.duration - duration) > 2.0:
                    errors.append(
                        f"{prefix} {video_id}: transcript duration {transcript.duration:.3f}s differs "
                        f"from media/manifest {duration:.3f}s"
                    )
            except Exception as exc:
                errors.append(f"{prefix} {video_id}: invalid transcript: {exc}")

        relevant_windows = record.get("relevant_windows", [])
        if source in {"qvhighlights", "custom", "custom_pseudo"} and not relevant_windows:
            errors.append(f"{prefix} {video_id}: source {source!r} requires relevant_windows")
        for window_index, window in enumerate(relevant_windows):
            if not isinstance(window, list) or len(window) != 2:
                errors.append(f"{prefix} {video_id}: annotation {window_index} must be [start, end]")
                continue
            start, end = map(float, window)
            if start < 0 or end <= start or (duration > 0 and end > duration + 1e-3):
                errors.append(
                    f"{prefix} {video_id}: annotation {window_index} is outside [0, {duration:.3f}]"
                )

        try:
            labels = create_window_labels(record, window_sec=window_sec, hop_sec=hop_sec)
        except Exception as exc:
            errors.append(f"{prefix} {video_id}: cannot create labels: {exc}")
            labels = []
        label_counts = Counter(item["label"] for item in labels)
        bucket = stats[split or "invalid"]
        bucket["videos"] += 1
        bucket["duration_seconds"] += max(duration, 0.0)
        bucket["positive_windows"] += label_counts["positive"]
        bucket["negative_windows"] += label_counts["negative"]
        bucket["ignored_windows"] += label_counts["ignored"]
        bucket["domains"][domain or "missing"] += 1
        bucket["sources"][source or "missing"] += 1
        if label_counts["positive"] and label_counts["negative"]:
            bucket["pairable_videos"] += 1
            bucket["pair_count"] += label_counts["positive"] * label_counts["negative"]
        valid_records.append(record)

    for split, bucket in stats.items():
        if split in {"train", "val", "test"}:
            if bucket["positive_windows"] == 0 or bucket["negative_windows"] == 0:
                errors.append(f"split {split!r} must contain both positive and negative windows")
            if bucket["pairable_videos"] == 0:
                errors.append(f"split {split!r} has no video with pairable positive/negative windows")

    present_domains = {record.get("domain") for record in records}
    expected_domains = required_domains or set()
    missing_domains = sorted(expected_domains - present_domains)
    if missing_domains:
        errors.append(f"manifest is missing required domains: {', '.join(missing_domains)}")
    elif present_domains != VALID_DOMAINS:
        warnings.append(
            "manifest is not cross-domain; present domains: "
            + ", ".join(sorted(str(item) for item in present_domains if item))
        )

    serializable_stats: dict[str, Any] = {}
    for split, bucket in sorted(stats.items()):
        serializable_stats[split] = {
            **{key: value for key, value in bucket.items() if key not in {"domains", "sources"}},
            "duration_seconds": round(bucket["duration_seconds"], 3),
            "domains": dict(sorted(bucket["domains"].items())),
            "sources": dict(sorted(bucket["sources"].items())),
        }
    try:
        l_ref = compute_lref(valid_records) if valid_records else None
    except Exception as exc:
        errors.append(f"cannot compute L_ref: {exc}")
        l_ref = None
    return {
        "valid": not errors,
        "manifest": str(manifest),
        "project_root": str(root),
        "video_count": len(records),
        "L_ref": l_ref,
        "splits": serializable_stats,
        "errors": errors,
        "warnings": warnings,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--report", default=None)
    parser.add_argument("--window-sec", type=float, default=5.0)
    parser.add_argument("--hop-sec", type=float, default=1.0)
    parser.add_argument("--minimum-videos", type=int, default=1)
    parser.add_argument(
        "--require-domain",
        action="append",
        choices=sorted(VALID_DOMAINS),
        default=[],
        help="Repeat to enforce one or more domains",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report = validate_manifest(
        args.manifest,
        project_root=args.project_root,
        window_sec=args.window_sec,
        hop_sec=args.hop_sec,
        minimum_videos=args.minimum_videos,
        required_domains=set(args.require_domain),
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = report_path.with_suffix(report_path.suffix + ".tmp")
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(report_path)
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
