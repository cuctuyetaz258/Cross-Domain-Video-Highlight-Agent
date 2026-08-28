"""Build canonical seven-channel LTR feature caches from a training manifest."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from highlight_agent.backend import load_transcript
from highlight_agent.features.acoustic import extract_windowed_acoustic_features
from highlight_agent.features.alignment import build_feature_matrix
from highlight_agent.features.visual_new import extract_gesture_signal, extract_scene_changes
from highlight_agent.models.train_offline import (
    FEATURE_SAMPLE_RATE,
    feature_cache_metadata,
    load_feature_matrix,
    load_training_manifest,
)
from scripts.validate_training_data import probe_video, resolve_record_path


def transcript_word_scores(transcript: Any) -> list[tuple[float, float, float]]:
    """Create deterministic TF-IDF density scores aligned to transcript words/segments."""

    segments = list(transcript.segments)
    if not segments:
        return []
    texts = [segment.text for segment in segments]
    try:
        matrix = TfidfVectorizer(lowercase=True, ngram_range=(1, 2)).fit_transform(texts)
        raw = np.asarray(matrix.mean(axis=1)).reshape(-1).astype(np.float32)
    except ValueError:
        raw = np.ones(len(segments), dtype=np.float32)
    minimum = float(raw.min(initial=0.0))
    maximum = float(raw.max(initial=0.0))
    scores = (raw - minimum) / (maximum - minimum) if maximum > minimum else np.ones_like(raw)
    aligned: list[tuple[float, float, float]] = []
    for segment, score in zip(segments, scores):
        if segment.words:
            aligned.extend((word.start, word.end, float(score)) for word in segment.words)
        else:
            aligned.append((segment.start, segment.end, float(score)))
    return aligned


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

    stage_times: dict[str, float] = {}
    stage_started = time.perf_counter()
    print(f"  {video_id}: acoustic", flush=True)
    acoustic, acoustic_windows = extract_windowed_acoustic_features(audio_path)
    stage_times["acoustic"] = time.perf_counter() - stage_started

    stage_started = time.perf_counter()
    print(f"  {video_id}: transcript TF-IDF", flush=True)
    word_scores = transcript_word_scores(transcript)
    stage_times["semantic"] = time.perf_counter() - stage_started

    stage_started = time.perf_counter()
    print(f"  {video_id}: scene detection", flush=True)
    scene_times = extract_scene_changes(video_path, duration) if include_scenes else []
    stage_times["scene"] = time.perf_counter() - stage_started

    stage_started = time.perf_counter()
    print(f"  {video_id}: gesture sampling", flush=True)
    gesture = (
        extract_gesture_signal(video_path, duration, sample_rate=2.0)
        if include_gesture
        else np.zeros(int(duration * 2.0), dtype=np.float32)
    )
    stage_times["gesture"] = time.perf_counter() - stage_started

    matrix = build_feature_matrix(
        acoustic,
        acoustic_windows,
        scene_times,
        gesture,
        word_scores,
        None,
        duration,
        sample_rate=FEATURE_SAMPLE_RATE,
    )
    metadata = {
        **feature_cache_metadata(video_id, matrix),
        "duration": duration,
        "domain": record["domain"],
        "source": record["source"],
        "split": record["split"],
        "label_protocol": record.get("label_protocol"),
        "video_path": str(record["video_path"]),
        "audio_path": str(record["audio_path"]),
        "transcript_path": str(record["transcript_path"]),
        "extractor": {
            "device": device,
            "acoustic_window_seconds": 30.0,
            "acoustic_hop_seconds": 30.0,
            "text_method": "segment_tfidf_density",
            "scene_enabled": include_scenes,
            "gesture_enabled": include_gesture,
            "gesture_sample_rate": 2.0,
            "interaction_method": "none_non_podcast",
        },
        "observations": {
            "acoustic_window_count": len(acoustic_windows),
            "text_interval_count": len(word_scores),
            "scene_count": len(scene_times),
            "gesture_sample_count": int(len(gesture)),
        },
        "stage_seconds": {key: round(value, 3) for key, value in stage_times.items()},
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
    limit: int | None = None,
    force: bool = False,
    include_scenes: bool = True,
    include_gesture: bool = True,
    device: str = "cpu",
) -> dict[str, Any]:
    records = load_training_manifest(manifest_path, split=split)
    if limit is not None:
        records = records[:limit]
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        print(f"[{index}/{len(records)}] extracting {record['video_id']}...", flush=True)
        try:
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
    return {
        "manifest": str(Path(manifest_path).resolve()),
        "output_dir": str(Path(output_dir).resolve()),
        "requested_video_count": len(records),
        "built_count": sum(item["status"] == "built" for item in results),
        "skipped_count": sum(item["status"] == "skipped_valid" for item in results),
        "failed_count": sum(item["status"] == "failed" for item in results),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "results": results,
    }


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
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--no-scenes", action="store_true", help="Debug only: zero the scene channel")
    parser.add_argument("--no-gesture", action="store_true", help="Debug only: zero the gesture channel")
    parser.add_argument("--report", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report = build_manifest_caches(
        args.manifest,
        project_root=args.project_root,
        output_dir=args.output_dir,
        split=args.split,
        limit=args.limit,
        force=args.force,
        include_scenes=not args.no_scenes,
        include_gesture=not args.no_gesture,
        device=args.device,
    )
    if args.report:
        _write_json(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if report["failed_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
