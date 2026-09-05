"""Build canonical seven-channel LTR feature caches from a training manifest."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from highlight_agent.backend import load_transcript  # noqa: E402
from highlight_agent.features.ltr_pipeline import build_ltr_features  # noqa: E402
from highlight_agent.features.semantic import transcript_tfidf_density_scores  # noqa: E402
from highlight_agent.features.visual_new import (  # noqa: E402
    extract_gesture_observation,
)
from highlight_agent.models.train_offline import (  # noqa: E402
    FEATURE_CHANNELS,
    feature_cache_metadata,
    load_feature_matrix,
    load_training_manifest,
)
from scripts.validate_training_data import probe_video, resolve_record_path  # noqa: E402


def transcript_word_scores(transcript: Any) -> list[tuple[float, float, float]]:
    """Giữ API script cũ nhưng dùng chung semantic transform với runtime"""

    return transcript_tfidf_density_scores(transcript)


def _atomic_write_cache(cache_dir: Path, matrix: np.ndarray, metadata: dict[str, Any]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = cache_dir / "feature_matrix.npy"
    matrix_temporary = cache_dir / "feature_matrix.npy.tmp"
    with matrix_temporary.open("wb") as handle:
        np.save(handle, matrix, allow_pickle=False)
    matrix_temporary.replace(matrix_path)
    metadata_path = cache_dir / "metadata.json"
    metadata_temporary = cache_dir / "metadata.json.tmp"
    metadata_temporary.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    metadata_temporary.replace(metadata_path)


def refresh_gesture_observation_for_record(
    record: dict[str, Any],
    *,
    project_root: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Cập nhật observation gesture mà không tính lại sáu channel còn lại"""

    started = time.perf_counter()
    video_id = str(record["video_id"])
    cache_root = Path(output_dir)
    matrix = load_feature_matrix(cache_root, video_id)
    cache_dir = cache_root / video_id
    metadata_path = cache_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    video_path = resolve_record_path(record["video_path"], project_root)
    duration = float(record.get("duration") or probe_video(video_path)[0])
    gesture_result = extract_gesture_observation(video_path, duration, sample_rate=2.0)

    metadata.setdefault("extractor", {}).update(
        {
            "gesture_enabled": True,
            "gesture_sample_rate": 2.0,
            "gesture_status": gesture_result.status,
        }
    )
    metadata.setdefault("observations", {}).update(
        {
            "gesture_sample_count": int(len(gesture_result.signal)),
            "gesture_decoded_sample_count": gesture_result.decoded_sample_count,
            "gesture_detected_sample_count": gesture_result.detected_sample_count,
        }
    )
    temporary = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    temporary.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(metadata_path)
    return {
        "video_id": video_id,
        "status": "refreshed_gesture_observation",
        "shape": list(matrix.shape),
        "gesture_status": gesture_result.status,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


def build_cache_for_record(
    record: dict[str, Any],
    *,
    project_root: str | Path,
    output_dir: str | Path,
    force: bool = False,
    include_scenes: bool = True,
    include_gesture: bool = True,
    device: str = "cpu",
) -> dict[str, Any]:
    """Extract production signals and atomically persist one canonical cache."""

    started = time.perf_counter()
    video_id = str(record["video_id"])
    cache_root = Path(output_dir)
    cache_dir = cache_root / video_id
    if not force:
        try:
            matrix = load_feature_matrix(cache_root, video_id)
            return {
                "video_id": video_id,
                "status": "skipped_valid",
                "shape": list(matrix.shape),
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }
        except (FileNotFoundError, ValueError):
            pass

    video_path = resolve_record_path(record["video_path"], project_root)
    audio_path = resolve_record_path(record["audio_path"], project_root)
    transcript_path = resolve_record_path(record["transcript_path"], project_root)
    duration = float(record.get("duration") or probe_video(video_path)[0])
    transcript = load_transcript(transcript_path)
    if abs(transcript.duration - duration) > 2.0:
        raise ValueError(
            f"transcript duration {transcript.duration:.3f}s differs from manifest {duration:.3f}s"
        )

    print(f"  {video_id}: unified LTR features", flush=True)
    bundle = build_ltr_features(
        video_path=video_path,
        audio_path=audio_path,
        transcript=transcript,
        domain=record["domain"],
        duration=duration,
        known_speaker_count=record.get("known_speaker_count"),
        min_speaker_count=record.get("min_speaker_count"),
        max_speaker_count=record.get("max_speaker_count"),
        include_scenes=include_scenes,
        include_gesture=include_gesture,
        device=device,
    )
    matrix = bundle.matrix
    metadata = {
        **feature_cache_metadata(video_id, matrix),
        "duration": duration,
        "domain": record["domain"],
        "source": record["source"],
        "dataset": record.get("dataset"),
        "category": record.get("category"),
        "split": record["split"],
        "label_protocol": record.get("label_protocol"),
        "video_path": str(record["video_path"]),
        "audio_path": str(record["audio_path"]),
        "transcript_path": str(record["transcript_path"]),
        "feature_contract": bundle.metadata["feature_contract"],
        "extractor": bundle.metadata["extractor"],
        "observations": bundle.metadata["observations"],
        "channel_stats": bundle.metadata["channel_stats"],
        "stage_seconds": bundle.metadata["stage_seconds"],
    }
    _atomic_write_cache(cache_dir, matrix, metadata)
    validated = load_feature_matrix(cache_root, video_id)
    return {
        "video_id": video_id,
        "status": "built",
        "shape": list(validated.shape),
        "cache_bytes": (cache_dir / "feature_matrix.npy").stat().st_size,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "stage_seconds": metadata["stage_seconds"],
    }


def build_manifest_caches(
    manifest_path: str | Path,
    *,
    project_root: str | Path,
    output_dir: str | Path,
    split: str | None = None,
    domain: str | None = None,
    limit: int | None = None,
    force: bool = False,
    include_scenes: bool = True,
    include_gesture: bool = True,
    device: str = "cpu",
    refresh_gesture_observation: bool = False,
) -> dict[str, Any]:
    records = load_training_manifest(manifest_path, split=split)
    if domain is not None:
        records = [record for record in records if record.get("domain") == domain]
    if limit is not None:
        records = records[:limit]
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        print(f"[{index}/{len(records)}] extracting {record['video_id']}...", flush=True)
        try:
            if refresh_gesture_observation:
                result = refresh_gesture_observation_for_record(
                    record,
                    project_root=project_root,
                    output_dir=output_dir,
                )
            else:
                result = build_cache_for_record(
                    record,
                    project_root=project_root,
                    output_dir=output_dir,
                    force=force,
                    include_scenes=include_scenes,
                    include_gesture=include_gesture,
                    device=device,
                )
        except Exception as exc:
            result = {"video_id": record.get("video_id"), "status": "failed", "error": str(exc)}
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)
    report = {
        "manifest": str(Path(manifest_path).resolve()),
        "output_dir": str(Path(output_dir).resolve()),
        "requested_video_count": len(records),
        "built_count": sum(item["status"] in {"built", "refreshed_gesture_observation"} for item in results),
        "skipped_count": sum(item["status"] == "skipped_valid" for item in results),
        "failed_count": sum(item["status"] == "failed" for item in results),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "results": results,
    }
    if not report["failed_count"] and not refresh_gesture_observation:
        report["distribution"] = summarize_feature_distribution(records, output_dir)
    return report


def summarize_feature_distribution(
    records: list[dict[str, Any]],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Aggregate per-channel distribution and extractor statuses by split."""

    groups: dict[str, list[dict[str, Any]]] = {"all": records}
    for split in sorted({str(record["split"]) for record in records}):
        groups[split] = [record for record in records if record["split"] == split]

    result: dict[str, Any] = {}
    for group_name, group_records in groups.items():
        accumulators = {
            channel: {
                "count": 0,
                "sum": 0.0,
                "sum_sq": 0.0,
                "zero_count": 0,
                "min": float("inf"),
                "max": float("-inf"),
                "nonzero_video_count": 0,
            }
            for channel in FEATURE_CHANNELS
        }
        gesture_statuses: Counter[str] = Counter()
        scene_statuses: Counter[str] = Counter()
        for record in group_records:
            video_id = str(record["video_id"])
            matrix = load_feature_matrix(output_dir, video_id)
            metadata_path = Path(output_dir) / video_id / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            gesture_statuses[str(metadata["extractor"]["gesture_status"])] += 1
            scene_statuses[str(metadata["extractor"]["scene_status"])] += 1
            for channel, values in zip(FEATURE_CHANNELS, matrix):
                accumulator = accumulators[channel]
                accumulator["count"] += int(values.size)
                accumulator["sum"] += float(values.sum(dtype=np.float64))
                accumulator["sum_sq"] += float(np.square(values, dtype=np.float64).sum())
                accumulator["zero_count"] += int(np.count_nonzero(values == 0.0))
                accumulator["min"] = min(accumulator["min"], float(values.min()))
                accumulator["max"] = max(accumulator["max"], float(values.max()))
                accumulator["nonzero_video_count"] += int(np.any(values != 0.0))

        channel_report: dict[str, Any] = {}
        for channel, accumulator in accumulators.items():
            count = accumulator["count"]
            mean = accumulator["sum"] / count
            variance = max(0.0, accumulator["sum_sq"] / count - mean**2)
            channel_report[channel] = {
                "min": accumulator["min"],
                "max": accumulator["max"],
                "mean": mean,
                "std": variance**0.5,
                "zero_ratio": accumulator["zero_count"] / count,
                "nonzero_video_count": accumulator["nonzero_video_count"],
                "video_count": len(group_records),
            }
        result[group_name] = {
            "channels": channel_report,
            "gesture_status_counts": dict(sorted(gesture_statuses.items())),
            "scene_status_counts": dict(sorted(scene_statuses.items())),
        }
    return result


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(destination)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-dir", default="data/features_cache")
    parser.add_argument("--split", choices=["train", "val", "test"], default=None)
    parser.add_argument("--domain", choices=["lecture", "podcast", "standup"], default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--no-scenes", action="store_true", help="Debug only: zero the scene channel")
    parser.add_argument("--no-gesture", action="store_true", help="Debug only: zero the gesture channel")
    parser.add_argument(
        "--refresh-gesture-observation",
        action="store_true",
        help="Update gesture status metadata without rebuilding feature matrices",
    )
    parser.add_argument("--report", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = _parse_args(argv)
    report = build_manifest_caches(
        args.manifest,
        project_root=args.project_root,
        output_dir=args.output_dir,
        split=args.split,
        domain=args.domain,
        limit=args.limit,
        force=args.force,
        include_scenes=not args.no_scenes,
        include_gesture=not args.no_gesture,
        device=args.device,
        refresh_gesture_observation=args.refresh_gesture_observation,
    )
    if args.report:
        _write_json(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if report["failed_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
