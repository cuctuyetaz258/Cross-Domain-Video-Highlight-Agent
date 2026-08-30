"""Create an all-data training manifest from a cross-validation manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="One complete cross-validation manifest")
    parser.add_argument("--output", required=True, help="All-data JSONL manifest to create")
    args = parser.parse_args()

    input_path = Path(args.input)
    records = [
        json.loads(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError("input manifest has no records")
    video_ids = [str(record["video_id"]) for record in records]
    if len(video_ids) != len(set(video_ids)):
        raise ValueError("input manifest must contain each video exactly once")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            record["split"] = "train"
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output_path), "video_count": len(records)}, indent=2))


if __name__ == "__main__":
    main()
