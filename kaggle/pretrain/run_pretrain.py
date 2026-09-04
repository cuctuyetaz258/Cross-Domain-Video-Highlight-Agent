"""Kaggle runner: derive TVSum+SumMe features and pretrain the V2 TCN-LTR.

The public video Dataset mounted below /kaggle/input is read-only. Derived
audio, Whisper transcripts, feature caches, and checkpoints are written only
to /kaggle/working and archived on a successful run.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

INPUT_ROOT = Path("/kaggle/input")
WORKING = Path("/kaggle/working/v2_pretrain_project")
RUN_MODE = "materialize"  # validate | materialize | train


def run(*args: str) -> None:
    subprocess.run(args, cwd=WORKING, check=True)


def find_one(filename: str) -> Path:
    matches = list(INPUT_ROOT.rglob(filename))
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one mounted {filename!r}, found {matches}")
    return matches[0]


def find_project_input() -> Path:
    manifest = find_one("tvsum_summe.jsonl")
    # The private input bundle stores manifests in data/manifests/.
    return manifest.parents[2]


def copy_project_input() -> None:
    source = find_project_input()
    for item in source.iterdir():
        if item.name == "dataset-metadata.json":
            continue
        if item.is_dir():
            shutil.copytree(item, WORKING / item.name)
        elif item.suffix in {".zip", ".tar"}:
            shutil.unpack_archive(item, WORKING)
        else:
            shutil.copy2(item, WORKING / item.name)


def prepare_media(include_transcription: bool) -> None:
    args = [
        "python",
        "scripts/prepare_kaggle_benchmark_manifest.py",
        "--source-manifest",
        "data/manifests/tvsum_summe.jsonl",
        "--media-root",
        str(INPUT_ROOT),
        "--derived-root",
        "data/raw/kaggle_benchmark/processed",
        "--output-manifest",
        "data/manifests/tvsum_summe_kaggle.jsonl",
    ]
    if include_transcription:
        args.append("--prepare-media")
    run(*args)


def archive_cache_artifacts() -> None:
    """Archive only reusable training inputs, never bulky derived media."""

    artifact = Path("/kaggle/working/v2_tcn_ltr_pretrain_cache")
    if artifact.exists():
        shutil.rmtree(artifact)
    (artifact / "data/manifests").mkdir(parents=True)
    (artifact / "data/reports").mkdir(parents=True)
    shutil.copy2(
        WORKING / "data/manifests/tvsum_summe_kaggle.jsonl",
        artifact / "data/manifests/tvsum_summe_kaggle.jsonl",
    )
    shutil.copytree(WORKING / "data/features_cache", artifact / "data/features_cache")
    shutil.copy2(
        WORKING / "data/reports/tcn_ltr_tvsum_summe_cache_report.json",
        artifact / "data/reports/tcn_ltr_tvsum_summe_cache_report.json",
    )
    shutil.make_archive(str(artifact), "zip", artifact)


if WORKING.exists():
    shutil.rmtree(WORKING)
WORKING.mkdir(parents=True)
copy_project_input()

if RUN_MODE == "validate":
    prepare_media(include_transcription=False)
    print("Benchmark media mapping passed. Set RUN_MODE='materialize' for derived artifacts.")
elif RUN_MODE == "materialize":
    prepare_media(include_transcription=True)
    run(
        "python", "scripts/build_feature_cache.py",
        "--manifest", "data/manifests/tvsum_summe_kaggle.jsonl",
        "--project-root", ".", "--output-dir", "data/features_cache", "--device", "cuda",
        "--report", "data/reports/tcn_ltr_tvsum_summe_cache_report.json",
    )
    run(
        "python", "scripts/validate_ltr_cache_manifest.py",
        "--manifest", "data/manifests/tvsum_summe_kaggle.jsonl", "--cache-dir", "data/features_cache", "--split", "train",
    )
    archive_cache_artifacts()
elif RUN_MODE == "train":
    # A successful materialize version must first be published as a private
    # Dataset and attached here; this avoids rebuilding 75 video caches.
    manifest = find_one("tvsum_summe_kaggle.jsonl")
    cache = manifest.parents[2] / "data/features_cache"
    if not cache.is_dir():
        raise FileNotFoundError(f"cache Dataset does not contain {cache}")
    run(
        "python", "-m", "highlight_agent.models.train_tcn_ltr",
        "--manifest", str(manifest), "--cache-dir", str(cache),
        "--output", "data/models/tcn_ltr_pretrained_tvsum_summe.pt",
        "--last-output", "data/models/tcn_ltr_pretrained_tvsum_summe_last.pt",
        "--training-log", "data/reports/tcn_ltr_pretrain_log.json",
        "--history-csv", "data/reports/tcn_ltr_pretrain_history.csv",
        "--training-plot", "data/reports/tcn_ltr_pretrain_curves.svg",
        "--max-epochs", "50", "--patience", "15", "--lr", "1e-4",
    )
    shutil.make_archive("/kaggle/working/v2_tcn_ltr_pretrain", "zip", WORKING / "data")
else:
    raise ValueError(f"unknown RUN_MODE: {RUN_MODE}")
