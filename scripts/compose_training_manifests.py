"""Compose canonical dataset manifests while rejecting video/split leakage."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from highlight_agent.models.train_offline import load_training_manifest


def compose_manifests(inputs: list[str | Path], output: str | Path) -> dict[str, Any]:
    if len(inputs) < 2:
        raise ValueError("at least two input manifests are required")
    records: list[dict[str, Any]] = []
    seen_video_ids: dict[str, str] = {}
    seen_media: dict[str, str] = {}
    for source_path in inputs:
        for record in load_training_manifest(source_path):
            video_id = str(record["video_id"])
            if video_id in seen_video_ids:
                raise ValueError(
                    f"video_id {video_id!r} appears in {seen_video_ids[video_id]} and {source_path}"
                )
            seen_video_ids[video_id] = str(source_path)
            media_path = str(record.get("video_path", "")).replace("\\", "/").lower()
            if media_path and media_path in seen_media:
                raise ValueError(
                    f"video_path {media_path!r} is shared by {seen_media[media_path]!r} and {video_id!r}"
                )
            if media_path:
                seen_media[media_path] = video_id
            if record.get("split") not in {"train", "val", "test"}:
                raise ValueError(f"video {video_id!r} has invalid split")
            records.append(record)
    records.sort(key=lambda record: (str(record["split"]), str(record["source"]), str(record["video_id"])))
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    temporary.replace(destination)
    return {
        "output": str(destination),
        "video_count": len(records),
        "splits": dict(sorted(Counter(str(record["split"]) for record in records).items())),
        "sources": dict(sorted(Counter(str(record["source"]) for record in records).items())),
        "domains": dict(sorted(Counter(str(record["domain"]) for record in records).items())),
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result = compose_manifests(args.input, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
