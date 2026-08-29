"""Prepare, cache, and evaluate the reproducible in-domain LTR pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from evaluation.evaluate_indomain import evaluate_single_video, load_ground_truth
from evaluation.evaluate_ltr import evaluate_manifest
from highlight_agent.backend import prepare_video
from highlight_agent.features.nms_topk import extract_topk_nms
from highlight_agent.features.overlap_blender import blend_scores
from highlight_agent.media.audio import probe_duration
from highlight_agent.media.transcript import save_transcript, transcribe_with_whisper
from highlight_agent.models.ltr_scorer import AdditiveAttentionScorer
from highlight_agent.models.train_offline import train
from highlight_agent.paths import portable_relative_path
from scripts.build_feature_cache import build_manifest_caches
from scripts.validate_training_data import probe_video, validate_manifest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def _load_catalog(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != "1.0":
        raise ValueError("unsupported in-domain pilot catalog schema")
    if len(payload.get("videos", [])) != 6 or len(payload.get("folds", [])) != 6:
        raise ValueError("pilot catalog must define exactly six videos and six folds")
    return payload


def _digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _record(video: dict[str, Any], split: str, media_root: Path) -> dict[str, Any]:
    video_id = str(video["video_id"])
    workspace = media_root / video_id
    duration = float(video["duration"])
    source_video = workspace / "source_video.mp4"
    if source_video.is_file():
        duration, _fps = probe_video(source_video)
    return {
        "video_id": video_id,
        "url": video["url"],
        "domain": video["domain"],
        "source": "in_domain_ordinal",
        "split": split,
        "duration": duration,
        "video_path": portable_relative_path(workspace / "source_video.mp4", PROJECT_ROOT),
        "audio_path": portable_relative_path(workspace / "audio.wav", PROJECT_ROOT),
        "transcript_path": portable_relative_path(workspace / "transcript.json", PROJECT_ROOT),
        "annotation_path": video["annotation_path"],
        **{
            key: int(video[key])
            for key in ("known_speaker_count", "min_speaker_count", "max_speaker_count")
            if key in video
        },
    }


def write_fold_manifests(catalog_path: str | Path, output_dir: str | Path) -> list[Path]:
    catalog = _load_catalog(catalog_path)
    videos = {str(video["video_id"]): video for video in catalog["videos"]}
    media_root = PROJECT_ROOT / str(catalog["media_root"])
    destination = Path(output_dir)
    paths: list[Path] = []
    for fold in catalog["folds"]:
        validation_id = str(fold["validation"])
        test_id = str(fold["test"])
        records = []
        for video_id, video in videos.items():
            split = "val" if video_id == validation_id else "test" if video_id == test_id else "train"
            records.append(_record(video, split, media_root))
        path = destination / f"fold_{fold['id']}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records), encoding="utf-8")
        paths.append(path)
    return paths


def prepare_media(
    catalog_path: str | Path,
    *,
    cookies_browser: str | None,
    report_path: str | Path,
    video_id: str | None = None,
) -> dict[str, Any]:
    catalog = _load_catalog(catalog_path)
    media_root = PROJECT_ROOT / str(catalog["media_root"])
    results: list[dict[str, Any]] = []
    videos = [video for video in catalog["videos"] if video_id is None or video["video_id"] == video_id]
    if not videos:
        raise ValueError(f"video_id {video_id!r} is not in the pilot catalog")
    for video in videos:
        video_id = str(video["video_id"])
        workspace = media_root / video_id
        required = [workspace / "source_video.mp4", workspace / "audio.wav", workspace / "transcript.json"]
        try:
            if all(path.is_file() for path in required):
                status = "reused"
            elif required[0].is_file() and required[1].is_file():
                transcript = transcribe_with_whisper(
                    required[1], video_id=video_id, duration=probe_duration(required[0]), chapters=[]
                )
                save_transcript(transcript, required[2])
                status = "transcribed_existing_media"
            else:
                prepare_video(video["url"], output_root=media_root, cookies_browser=cookies_browser, transcript_source="whisper")
                status = "prepared"
            item = {
                "video_id": video_id,
                "url": video["url"],
                "status": status,
                "files": {path.name: _digest(path) for path in required if path.is_file()},
            }
        except Exception as exc:
            item = {"video_id": video_id, "url": video["url"], "status": "failed", "error": str(exc)}
        results.append(item)
        report = {"catalog": str(Path(catalog_path).resolve()), "media_root": str(media_root), "results": results}
        _write_json(Path(report_path), report)
    report = {"catalog": str(Path(catalog_path).resolve()), "media_root": str(media_root), "results": results}
    return report


def cache_media(manifest_dir: str | Path, cache_dir: str | Path, report_path: str | Path) -> dict[str, Any]:
    manifests = sorted(Path(manifest_dir).glob("fold_*.jsonl"))
    if not manifests:
        raise ValueError("no fold manifests found")
    # Every fold references all six videos; build the cache once from the first manifest.
    validation = validate_manifest(manifests[0], project_root=PROJECT_ROOT)
    if not validation["valid"]:
        raise ValueError("pilot manifest validation failed: " + "; ".join(validation["errors"]))
    # PySceneDetect imports PyAV, which conflicts with OpenCV FFmpeg dylibs on
    # several macOS environments. The OpenCV fallback retains a scene channel.
    os.environ.setdefault("HIGHLIGHT_AGENT_SCENE_BACKEND", "opencv")
    report = build_manifest_caches(
        manifests[0],
        project_root=PROJECT_ROOT,
        output_dir=cache_dir,
        device="cpu",
    )
    report["validation"] = validation
    _write_json(Path(report_path), report)
    return report


def _candidate_temporal_metrics(manifest: Path, cache_dir: str | Path, checkpoint: str | Path) -> dict[str, Any]:
    """Evaluate NMS candidates against contiguous annotation regions scored >= 4."""

    import numpy as np
    import torch

    from highlight_agent.models.train_offline import build_window_examples, load_feature_matrix, load_training_manifest

    records = load_training_manifest(manifest, split="test")
    model, metadata = AdditiveAttentionScorer.load_checkpoint(checkpoint, device="cpu")
    per_video: list[dict[str, Any]] = []
    for record in records:
        examples = build_window_examples(cache_dir, [record])
        features = torch.as_tensor(np.stack([example.feature for example in examples]), dtype=torch.float32)
        with torch.no_grad():
            scores = model(features).reshape(-1).numpy()
        matrix = load_feature_matrix(cache_dir, str(record["video_id"]))
        candidates = extract_topk_nms(
            blend_scores(scores, T=matrix.shape[1]), k=3, reference_duration=float(metadata["L_ref"])
        )
        ground_truth = load_ground_truth(record["annotation_path"])
        predictions = [candidate.model_dump() for candidate in candidates]
        per_video.append(
            {
                "video_id": record["video_id"],
                "candidates": predictions,
                **evaluate_single_video(predictions, ground_truth["highlights"], k=3),
            }
        )
    metric_names = ("hit@1_iou0.3", "hit@3_iou0.3", "hit@3_iou0.5", "f1@3", "mean_iou")
    return {
        "per_video": per_video,
        "macro": {name: float(statistics.mean(row[name] for row in per_video)) for name in metric_names},
    }


def run_folds(
    manifest_dir: str | Path,
    cache_dir: str | Path,
    init_checkpoint: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    output = Path(output_root)
    manifests = sorted(Path(manifest_dir).glob("fold_*.jsonl"))
    if len(manifests) != 6:
        raise ValueError("expected six generated fold manifests")
    results: list[dict[str, Any]] = []
    best_epochs: list[int] = []
    for manifest in manifests:
        fold_id = manifest.stem.removeprefix("fold_")
        fold_output = output / f"fold_{fold_id}"
        frozen = evaluate_manifest(
            manifest=manifest, cache_dir=cache_dir, checkpoint=init_checkpoint, split="test", device="auto", profiles=[], top_k=3
        )
        frozen["temporal_candidate_metrics"] = _candidate_temporal_metrics(manifest, cache_dir, init_checkpoint)
        model_path = PROJECT_ROOT / "data/models/in_domain_pilot" / f"fold_{fold_id}.pt"
        log_path = fold_output / "training_log.json"
        from highlight_agent.models.train_offline import load_training_manifest

        train_records = load_training_manifest(manifest, split="train")
        val_records = load_training_manifest(manifest, split="val")
        train(
            cache_dir, train_records, model_path, val_records,
            init_checkpoint_path=init_checkpoint, lr=1e-4, max_epochs=20, patience=5,
            seed=42, training_log_path=log_path, last_checkpoint_path=fold_output / "last.pt",
        )
        fine_tuned = evaluate_manifest(
            manifest=manifest, cache_dir=cache_dir, checkpoint=model_path, split="test", device="auto", profiles=[], top_k=3
        )
        fine_tuned["temporal_candidate_metrics"] = _candidate_temporal_metrics(manifest, cache_dir, model_path)
        training_log = json.loads(log_path.read_text(encoding="utf-8"))
        best_epochs.append(int(training_log["best_epoch"]))
        result = {"fold_id": fold_id, "frozen": frozen, "fine_tuned": fine_tuned, "training": training_log}
        _write_json(fold_output / "result.json", result)
        results.append(result)
    metric_keys = (
        "average_precision",
        "kendall_tau",
        "spearman_rho",
        "window_f1_at_positive_count",
        "positive_hit_at_k",
    )
    summaries: dict[str, dict[str, dict[str, float]]] = {}
    for name in ("frozen", "fine_tuned"):
        values = [result[name]["methods"][0] for result in results]
        summaries[name] = {
            metric: {
                "mean": float(statistics.mean(float(item[metric]) for item in values)),
                "stddev": float(statistics.stdev(float(item[metric]) for item in values)),
            }
            for metric in metric_keys
        }
    aggregate = {
        "fold_count": len(results),
        "median_best_epoch": int(statistics.median(best_epochs)),
        "best_epochs": best_epochs,
        "metric_summary": summaries,
        "folds": results,
    }
    _write_json(output / "aggregate.json", aggregate)
    return aggregate


def train_operational(
    manifest_dir: str | Path,
    cache_dir: str | Path,
    init_checkpoint: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Train a post-evaluation operational artifact, never a held-out result."""

    aggregate_path = Path(output_root) / "aggregate.json"
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    epochs = int(aggregate["median_best_epoch"])
    manifest = sorted(Path(manifest_dir).glob("fold_*.jsonl"))[0]
    from highlight_agent.models.train_offline import load_training_manifest

    records = load_training_manifest(manifest)
    output_path = PROJECT_ROOT / "data/models/in_domain_pilot/all_data_operational.pt"
    log_path = Path(output_root) / "operational_training_log.json"
    train(
        cache_dir,
        records,
        output_path,
        init_checkpoint_path=init_checkpoint,
        lr=1e-4,
        max_epochs=epochs,
        patience=epochs,
        seed=42,
        training_log_path=log_path,
        last_checkpoint_path=Path(output_root) / "operational_last.pt",
    )
    result = {
        "role": "operational_after_cross_validation",
        "not_a_held_out_evaluation": True,
        "epochs": epochs,
        "checkpoint": str(output_path),
        "training_log": str(log_path),
    }
    _write_json(Path(output_root) / "operational.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["prepare", "write-folds", "cache", "run-folds", "train-operational"]
    )
    parser.add_argument("--catalog", default="data/manifests/in_domain_pilot.json")
    parser.add_argument("--fold-dir", default="data/reports/in_domain_pilot/folds")
    parser.add_argument("--cache-dir", default="data/features_cache/in_domain_pilot")
    parser.add_argument("--init-checkpoint", default="data/models/ltr_scorer.pt")
    parser.add_argument("--output-dir", default="data/reports/in_domain_pilot")
    parser.add_argument("--cookies-browser", default=None)
    parser.add_argument("--video-id", default=None, help="Prepare one catalog video per process to avoid native decoder leaks")
    args = parser.parse_args()

    if args.command == "prepare":
        payload = prepare_media(
            args.catalog,
            cookies_browser=args.cookies_browser,
            report_path=Path(args.output_dir) / "preparation.json",
            video_id=args.video_id,
        )
    elif args.command == "write-folds":
        payload = {"manifests": [str(path) for path in write_fold_manifests(args.catalog, args.fold_dir)]}
    elif args.command == "cache":
        payload = cache_media(args.fold_dir, args.cache_dir, Path(args.output_dir) / "cache.json")
    elif args.command == "run-folds":
        payload = run_folds(args.fold_dir, args.cache_dir, args.init_checkpoint, args.output_dir)
    else:
        payload = train_operational(args.fold_dir, args.cache_dir, args.init_checkpoint, args.output_dir)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
