"""Kaggle runner: derive TVSum+SumMe features and pretrain the V2 TCN-LTR.

The public video Dataset mounted below /kaggle/input is read-only. Derived
audio, Whisper transcripts, feature caches, and checkpoints are written only
to /kaggle/working and archived on a successful run.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

INPUT_ROOT = Path("/kaggle/input")
WORKING = Path("/kaggle/working/v2_pretrain_project")
EXTRACTED_INPUT_ROOT = Path("/kaggle/working/v2_pretrain_input_archives")
RUN_MODE = "materialize"  # validate | materialize | train


def run(*args: str) -> None:
    subprocess.run(args, cwd=WORKING, check=True)


def install_materialization_dependencies() -> None:
    """Kaggle's base image omits the speech and visual extractors we require."""

    run(
        sys.executable,
        "-m",
        "pip",
        "install",
        "--quiet",
        "faster-whisper>=1,<2",
        "scenedetect>=0.6.4",
        # Newer MediaPipe builds removed the legacy ``solutions`` API used by
        # the reproducible seven-feature extractor.
        "mediapipe==0.10.14",
    )


def verify_materialization_dependencies() -> None:
    """Fail before 75-video extraction if the required legacy API is absent."""

    import mediapipe as mp

    if not hasattr(mp, "solutions"):
        raise RuntimeError(
            "incompatible MediaPipe: the V2 gesture extractor requires "
            "mediapipe.solutions; expected mediapipe==0.10.14"
        )


def load_optional_hf_token() -> None:
    """Expose an opt-in Kaggle Secret to Hugging Face dependent extractors."""

    try:
        from kaggle_secrets import UserSecretsClient

        token = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        print("HF_TOKEN secret is unavailable; continuing with anonymous Hugging Face downloads.")
        return
    if token:
        os.environ["HF_TOKEN"] = token
        print("HF_TOKEN secret loaded for authenticated Hugging Face downloads.")


def expand_input_archives() -> Path:
    """Expand Kaggle Dataset directory uploads when they arrive as archives."""

    if EXTRACTED_INPUT_ROOT.exists():
        return EXTRACTED_INPUT_ROOT
    EXTRACTED_INPUT_ROOT.mkdir(parents=True)
    for archive in INPUT_ROOT.rglob("*"):
        if archive.is_file() and archive.suffix.lower() in {".zip", ".tar"}:
            shutil.unpack_archive(archive, EXTRACTED_INPUT_ROOT)
    return EXTRACTED_INPUT_ROOT


def find_one(filename: str) -> Path:
    matches = list(INPUT_ROOT.rglob(filename))
    if not matches:
        matches = list(expand_input_archives().rglob(filename))
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


def find_benchmark_media_input() -> Path:
    """Locate the mounted Dataset root containing the two benchmark trees."""

    candidates = {
        path.parent
        for path in INPUT_ROOT.rglob("tvsum")
        if path.is_dir() and (path.parent / "summe" / "videos").is_dir() and (path / "videos").is_dir()
    }
    if len(candidates) != 1:
        raise RuntimeError(f"expected one benchmark media Dataset root, found {sorted(map(str, candidates))}")
    return candidates.pop()


def copy_benchmark_media_into_project() -> Path:
    """Make media real project files so manifest paths pass portability checks."""

    destination = WORKING / "data/raw/kaggle_benchmark/media"
    shutil.copytree(find_benchmark_media_input(), destination)
    return destination


def prepare_media(include_transcription: bool) -> None:
    args = [
        "python",
        "scripts/prepare_kaggle_benchmark_manifest.py",
        "--source-manifest",
        "data/manifests/tvsum_summe.jsonl",
        "--media-root", "data/raw/kaggle_benchmark/media",
        "--project-root", ".",
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
    load_optional_hf_token()
    install_materialization_dependencies()
    verify_materialization_dependencies()
    copy_benchmark_media_into_project()
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
