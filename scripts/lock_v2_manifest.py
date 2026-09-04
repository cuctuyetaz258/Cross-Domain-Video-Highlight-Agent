"""Create a reproducible fingerprint lock for the existing 10-video V2 protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_lock(manifest_dir: Path, cache_dir: Path, output: Path) -> dict[str, Any]:
    manifests = sorted(manifest_dir.glob("custom_fold[0-9].jsonl"))
    if len(manifests) != 5:
        raise ValueError("expected exactly five custom_fold*.jsonl manifests")
    records_by_video: dict[str, dict[str, Any]] = {}
    manifest_hashes = {}
    for manifest in manifests:
        manifest_hashes[manifest.name] = _sha256(manifest)
        for record in _records(manifest):
            video_id = str(record["video_id"])
            existing = records_by_video.setdefault(video_id, record)
            if existing.get("domain") != record.get("domain"):
                raise ValueError(f"inconsistent domain for {video_id}")
    if len(records_by_video) != 10:
        raise ValueError(f"expected 10 distinct videos, found {len(records_by_video)}")
    videos = []
    for video_id, record in sorted(records_by_video.items()):
        cache = cache_dir / video_id
        matrix, metadata = cache / "feature_matrix.npy", cache / "metadata.json"
        if not matrix.is_file() or not metadata.is_file():
            raise FileNotFoundError(f"missing feature cache for {video_id}")
        parsed_metadata = json.loads(metadata.read_text(encoding="utf-8"))
        if parsed_metadata.get("schema_version") != "1.1":
            raise ValueError(f"{video_id} cache schema is not 1.1")
        videos.append(
            {
                "video_id": video_id,
                "domain": record.get("domain"),
                "duration": record.get("duration"),
                "annotation_path": record.get("annotation_path"),
                "cache_matrix_sha256": _sha256(matrix),
                "cache_metadata_sha256": _sha256(metadata),
            }
        )
    lock = {
        "protocol": "v2_10video_video_disjoint_5fold",
        "feature_schema_version": "1.1",
        "source_manifests": manifest_hashes,
        "videos": videos,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    return lock


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", default="data/manifests")
    parser.add_argument("--cache-dir", default="data/features_cache")
    parser.add_argument("--output", default="data/manifests/v2_10video_lock.json")
    args = parser.parse_args()
    lock = build_lock(Path(args.manifest_dir), Path(args.cache_dir), Path(args.output))
    print(f"locked {len(lock['videos'])} videos in {args.output}")


if __name__ == "__main__":
    main()
