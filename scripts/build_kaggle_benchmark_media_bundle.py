"""Stage only benchmark videos needed by the V2 Kaggle pretraining manifest.

Files are hard-linked rather than copied, so staging consumes no additional
media storage on the local filesystem.  The resulting directory is intended
for a private Kaggle Dataset, not Git.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def stage_media(records: list[dict], output: Path) -> list[Path]:
    """Hard-link the exact video files referenced by a benchmark manifest."""

    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.mkdir(parents=True)
    staged: list[Path] = []
    seen: set[Path] = set()
    for record in records:
        source = Path(str(record["video_path"])).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"missing source media for {record['video_id']}: {source}")
        if source in seen:
            raise ValueError(f"manifest references duplicate media: {source}")
        seen.add(source)
        destination = output / str(record["source"]) / "videos" / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.hardlink_to(source)
        staged.append(destination)
    return staged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/manifests/tvsum_summe.jsonl")
    parser.add_argument("--output", default="tmp/kaggle_v2_benchmark_media")
    parser.add_argument("--dataset-id", default="nguyentrann0703/video-highlight-v2-benchmark-media")
    args = parser.parse_args()

    output = (ROOT / args.output).resolve()
    staged = stage_media(load_records((ROOT / args.manifest).resolve()), output)
    metadata = {
        "title": "Video Highlight V2 Benchmark Media",
        "id": args.dataset_id,
        "licenses": [{"name": "other"}],
    }
    (output / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "media_count": len(staged)}, sort_keys=True))


if __name__ == "__main__":
    main()
