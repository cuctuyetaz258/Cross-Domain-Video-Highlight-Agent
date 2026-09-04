"""Fail fast with every missing/incompatible cache required by an LTR manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_manifest(path: Path, split: str) -> list[dict]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if "video_id" not in record:
            raise ValueError(f"manifest line {line_number} is missing video_id")
        if record.get("split") == split:
            records.append(record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cache-dir", default="data/features_cache")
    parser.add_argument("--split", default="train")
    args = parser.parse_args()
    records = _load_manifest(Path(args.manifest), args.split)
    cache_dir = Path(args.cache_dir)
    issues: list[str] = []
    for record in records:
        video_id = str(record["video_id"])
        root = cache_dir / video_id
        matrix, metadata = root / "feature_matrix.npy", root / "metadata.json"
        if not matrix.is_file() or not metadata.is_file():
            issues.append(f"{video_id}: missing feature_matrix.npy or metadata.json")
            continue
        try:
            parsed = json.loads(metadata.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            issues.append(f"{video_id}: metadata.json is invalid JSON")
            continue
        if parsed.get("schema_version") != "1.1":
            issues.append(f"{video_id}: schema_version={parsed.get('schema_version')!r}, expected '1.1'")
    if issues:
        print("LTR cache validation failed:")
        print("\n".join(f"- {issue}" for issue in issues))
        raise SystemExit(1)
    print(f"validated {len(records)} {args.split} records against schema 1.1")


if __name__ == "__main__":
    main()
