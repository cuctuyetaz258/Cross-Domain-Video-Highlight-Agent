"""Audit ActionFormer artifacts and create leakage-safe five-fold manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def load_importance(path: Path) -> list[tuple[float, float, float]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [
        (float(row["start_sec"]), float(row["end_sec"]), float(row["importance"]))
        for row in rows
    ]


def interval_utility(
    start: float,
    end: float,
    importance: Iterable[tuple[float, float, float]],
) -> float:
    weighted = 0.0
    covered = 0.0
    for item_start, item_end, value in importance:
        overlap = max(0.0, min(end, item_end) - max(start, item_start))
        weighted += overlap * value
        covered += overlap
    return weighted / covered if covered else 0.0


def normalize_boundary(
    start: float,
    end: float,
    *,
    video_duration: float,
    importance: list[tuple[float, float, float]],
    min_duration: float = 30.0,
    max_duration: float = 90.0,
) -> tuple[float, float, str]:
    if not 0 <= start < end <= video_duration + 0.15:
        raise ValueError("boundary must lie inside the annotated video duration")
    duration = end - start
    if min_duration <= duration <= max_duration:
        return start, end, "unchanged"
    if video_duration < min_duration:
        raise ValueError("video is shorter than the minimum highlight duration")
    if duration < min_duration:
        center = (start + end) / 2
        normalized_start = min(max(0.0, center - min_duration / 2), video_duration - min_duration)
        return normalized_start, normalized_start + min_duration, "expanded_to_min"

    latest_start = end - max_duration
    candidate_starts = {start, latest_start}
    candidate_starts.update(
        max(start, min(latest_start, item_start))
        for item_start, item_end, _ in importance
        if item_end > start and item_start < end
    )
    normalized_start = max(
        sorted(candidate_starts),
        key=lambda value: (interval_utility(value, value + max_duration, importance), -value),
    )
    return normalized_start, normalized_start + max_duration, "cropped_to_best_max_window"


def assign_stratified_folds(
    records: list[dict[str, Any]],
    *,
    folds: int = 5,
    seed: int = 42,
) -> dict[int, list[dict[str, Any]]]:
    if folds < 3:
        raise ValueError("folds must be at least 3")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["domain"])].append(record)
    test_fold_by_id: dict[str, int] = {}
    for domain, domain_records in sorted(grouped.items()):
        ordered = sorted(domain_records, key=lambda item: str(item["video_id"]))
        random.Random(f"{seed}:{domain}").shuffle(ordered)
        for index, record in enumerate(ordered):
            test_fold_by_id[str(record["video_id"])] = index % folds

    manifests: dict[int, list[dict[str, Any]]] = {}
    for fold in range(folds):
        validation_fold = (fold + 1) % folds
        assigned: list[dict[str, Any]] = []
        for record in records:
            test_fold = test_fold_by_id[str(record["video_id"])]
            split = "test" if test_fold == fold else "val" if test_fold == validation_fold else "train"
            assigned.append(
                {
                    **record,
                    "fold": fold,
                    "split": split,
                    "split_seed": seed,
                    "test_fold": test_fold,
                }
            )
        manifests[fold] = sorted(assigned, key=lambda item: str(item["video_id"]))
    return manifests


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def audit_records(
    *,
    project_root: Path,
    boundary_dir: Path,
    importance_dir: Path,
    media_root: Path,
    feature_cache_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    issues: list[dict[str, str]] = []
    normalization_counts: Counter[str] = Counter()
    for boundary_path in sorted(boundary_dir.glob("*.json")):
        boundary = json.loads(boundary_path.read_text(encoding="utf-8"))
        video_id = str(boundary["video_id"])
        importance_path = importance_dir / f"{video_id}.csv"
        media_dir = media_root / video_id
        feature_dir = feature_cache_root / video_id
        paths = {
            "importance": importance_path,
            "video": media_dir / "source_video.mp4",
            "audio": media_dir / "audio.wav",
            "transcript": media_dir / "transcript.json",
            "feature": feature_dir / "feature_matrix.npy",
            "feature_metadata": feature_dir / "metadata.json",
        }
        missing = [name for name, path in paths.items() if not path.is_file()]
        if not importance_path.is_file():
            issues.append({"video_id": video_id, "issue": "missing importance CSV"})
            importance: list[tuple[float, float, float]] = []
            video_duration = max(float(item["end_time"]) for item in boundary["highlights"])
        else:
            importance = load_importance(importance_path)
            video_duration = max(end for _, end, _ in importance)
        normalized: list[dict[str, Any]] = []
        for highlight in boundary["highlights"]:
            start, end, policy = normalize_boundary(
                float(highlight["start_time"]),
                min(float(highlight["end_time"]), video_duration),
                video_duration=video_duration,
                importance=importance,
            )
            normalization_counts[policy] += 1
            normalized.append(
                {
                    "highlight_id": highlight["highlight_id"],
                    "original_start": float(highlight["start_time"]),
                    "original_end": float(highlight["end_time"]),
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "normalization": policy,
                    "importance": float(highlight.get("importance", 5)),
                }
            )
        artifact_ready = not missing
        if missing:
            issues.append({"video_id": video_id, "issue": "missing artifacts: " + ", ".join(missing)})
        records.append(
            {
                "video_id": video_id,
                "dataset": "custom_lecture_podcast",
                "source": "custom_actionformer",
                "domain": boundary["domain"],
                "duration": video_duration,
                "boundary_path": _relative(boundary_path, project_root),
                "importance_path": _relative(importance_path, project_root),
                "video_path": _relative(paths["video"], project_root),
                "audio_path": _relative(paths["audio"], project_root),
                "transcript_path": _relative(paths["transcript"], project_root),
                "feature_path": _relative(paths["feature"], project_root),
                "feature_metadata_path": _relative(paths["feature_metadata"], project_root),
                "artifact_ready": artifact_ready,
                "missing_artifacts": missing,
                "normalization_policy_version": "duration_30_90_v1",
                "highlights": normalized,
            }
        )
    fingerprint_payload = [
        {
            "video_id": item["video_id"],
            "domain": item["domain"],
            "duration": item["duration"],
            "highlights": item["highlights"],
        }
        for item in records
    ]
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    audit = {
        "schema_version": "1.0",
        "record_count": len(records),
        "ready_count": sum(bool(item["artifact_ready"]) for item in records),
        "domain_counts": dict(sorted(Counter(item["domain"] for item in records).items())),
        "highlight_count": sum(len(item["highlights"]) for item in records),
        "normalization_counts": dict(sorted(normalization_counts.items())),
        "dataset_fingerprint": fingerprint,
        "issues": issues,
    }
    return records, audit


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _atomic_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    temporary.replace(path)


def prepare_actionformer_data(
    *,
    project_root: str | Path,
    boundary_dir: str | Path,
    importance_dir: str | Path,
    media_root: str | Path,
    feature_cache_root: str | Path,
    manifest_dir: str | Path,
    audit_path: str | Path,
    folds: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    records, audit = audit_records(
        project_root=root,
        boundary_dir=Path(boundary_dir),
        importance_dir=Path(importance_dir),
        media_root=Path(media_root),
        feature_cache_root=Path(feature_cache_root),
    )
    manifests = assign_stratified_folds(records, folds=folds, seed=seed)
    manifest_paths: list[str] = []
    for fold, manifest in manifests.items():
        path = Path(manifest_dir) / f"actionformer_fold{fold}.jsonl"
        _atomic_jsonl(path, manifest)
        manifest_paths.append(str(path))
    audit["folds"] = {
        str(fold): dict(sorted(Counter(item["split"] for item in manifest).items()))
        for fold, manifest in manifests.items()
    }
    audit["manifests"] = manifest_paths
    _atomic_json(Path(audit_path), audit)
    return audit


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--boundary-dir", default="data/annotations/boundaries")
    parser.add_argument("--importance-dir", default="data/annotations/raw")
    parser.add_argument("--media-root", default="data/raw/in_domain_pilot")
    parser.add_argument("--feature-cache-root", default="data/features_cache")
    parser.add_argument("--manifest-dir", default="data/manifests")
    parser.add_argument("--audit-path", default="data/reports/actionformer_data_audit.json")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    audit = prepare_actionformer_data(
        project_root=args.project_root,
        boundary_dir=args.boundary_dir,
        importance_dir=args.importance_dir,
        media_root=args.media_root,
        feature_cache_root=args.feature_cache_root,
        manifest_dir=args.manifest_dir,
        audit_path=args.audit_path,
        folds=args.folds,
        seed=args.seed,
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if audit["ready_count"] == audit["record_count"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
