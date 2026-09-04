"""Stage the small, reproducible V2 fine-tuning input for a private Kaggle Dataset."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _copy(source: Path, destination: Path) -> None:
    if source.is_dir():
        shutil.copytree(source, destination, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="tmp/kaggle_v2_inputs")
    parser.add_argument("--dataset-id", default="nguyentrann0703/video-highlight-v2-inputs")
    args = parser.parse_args()
    output = ROOT / args.output
    if output.exists():
        raise FileExistsError(f"output already exists; choose a new --output path: {output}")
    output.mkdir(parents=True)

    # Only packages needed by V2 train/evaluation are bundled; no raw media or secrets.
    _copy(ROOT / "highlight_agent", output / "highlight_agent")
    _copy(ROOT / "evaluation", output / "evaluation")
    _copy(
        ROOT / "scripts/validate_ltr_cache_manifest.py",
        output / "scripts/validate_ltr_cache_manifest.py",
    )
    _copy(ROOT / "scripts/build_feature_cache.py", output / "scripts/build_feature_cache.py")
    _copy(
        ROOT / "scripts/prepare_kaggle_benchmark_manifest.py",
        output / "scripts/prepare_kaggle_benchmark_manifest.py",
    )
    _copy(ROOT / "scripts/validate_training_data.py", output / "scripts/validate_training_data.py")
    _copy(ROOT / "data/manifests", output / "data/manifests")
    for cache in sorted((ROOT / "data/features_cache").iterdir()):
        if cache.is_dir() and (cache / "metadata.json").is_file():
            metadata = json.loads((cache / "metadata.json").read_text(encoding="utf-8"))
            if metadata.get("schema_version") == "1.1":
                _copy(cache, output / "data/features_cache" / cache.name)
    (output / "README.md").write_text(
        "# Video Highlight V2 Inputs\n\n"
        "Private Kaggle input bundle for V2 TCN-LTR custom fine-tuning. "
        "It contains code, manifests, and schema-1.1 feature caches only; "
        "it intentionally excludes raw video, audio, transcripts, API keys, and model outputs.\n",
        encoding="utf-8",
    )
    (output / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": "Video Highlight V2 Inputs",
                "id": args.dataset_id,
                "licenses": [{"name": "other"}],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
