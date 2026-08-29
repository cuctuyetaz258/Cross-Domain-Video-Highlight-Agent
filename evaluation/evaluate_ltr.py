"""Compare the trained LTR scorer with static profile-weight baselines."""

from __future__ import annotations

import argparse
import csv
import json
import time
import tracemalloc
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from evaluation.metrics import compute_correlation
from highlight_agent.features.nms_topk import extract_topk_nms
from highlight_agent.features.overlap_blender import blend_scores
from highlight_agent.features.scoring import PROFILE_WEIGHTS
from highlight_agent.models.ltr_scorer import AdditiveAttentionScorer
from highlight_agent.models.train_offline import (
    FEATURE_CHANNELS,
    WindowExample,
    build_window_examples,
    load_feature_matrix,
    load_training_manifest,
)

PROFILE_NAMES = tuple(PROFILE_WEIGHTS)
CHANNEL_INDEX = {name: index for index, name in enumerate(FEATURE_CHANNELS)}


def profile_weight_scores(features: np.ndarray, profile: str) -> np.ndarray:
    """Project seven canonical cache channels onto the four runtime signal groups."""

    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != len(FEATURE_CHANNELS):
        raise ValueError(f"features must have shape (N, {len(FEATURE_CHANNELS)})")
    if profile not in PROFILE_WEIGHTS:
        raise ValueError(f"unknown profile: {profile}")

    rms = values[:, CHANNEL_INDEX["rms"]]
    pitch = values[:, CHANNEL_INDEX["pitch"]]
    silence = values[:, CHANNEL_INDEX["silence"]]
    signals = {
        "semantic": values[:, CHANNEL_INDEX["text_score"]],
        "acoustic": 0.45 * rms + 0.25 * pitch + 0.30 * (1.0 - silence),
        "interaction": values[:, CHANNEL_INDEX["turn_rate"]],
        "visual": 0.50 * (values[:, CHANNEL_INDEX["scene_change"]] + values[:, CHANNEL_INDEX["gesture"]]),
    }
    weights = PROFILE_WEIGHTS[profile]
    return np.asarray(
        sum(float(weights[name]) * signal for name, signal in signals.items()),
        dtype=np.float32,
    )


def window_metrics(
    examples: Iterable[WindowExample],
    scores: np.ndarray,
    *,
    top_k: int,
) -> dict[str, float | int | None]:
    """Compute ranking and explicitly window-level classification metrics."""

    rows = list(examples)
    predictions = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(rows) != len(predictions):
        raise ValueError("example and score counts do not match")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    continuous_targets = np.asarray([row.score for row in rows], dtype=np.float64)
    tau, _ = compute_correlation(predictions, continuous_targets, method="kendall")
    rho, _ = compute_correlation(predictions, continuous_targets, method="spearman")

    labels = np.asarray([row.label for row in rows], dtype=np.int64)
    labeled_mask = np.isin(labels, [0, 1])
    labeled_scores = predictions[labeled_mask]
    labeled_targets = labels[labeled_mask]
    positive_count = int(np.sum(labeled_targets == 1))

    average_precision: float | None = None
    window_f1: float | None = None
    hit_at_k: float | None = None
    if len(np.unique(labeled_targets)) == 2:
        average_precision = float(average_precision_score(labeled_targets, labeled_scores))
        selected = np.zeros(len(labeled_targets), dtype=np.int64)
        if positive_count:
            selected[np.argsort(labeled_scores)[-positive_count:]] = 1
            true_positives = int(np.sum((selected == 1) & (labeled_targets == 1)))
            precision = true_positives / positive_count
            recall = true_positives / positive_count
            window_f1 = float(2 * precision * recall / (precision + recall)) if true_positives else 0.0
        k = min(top_k, len(labeled_targets))
        top_indices = np.argsort(labeled_scores)[-k:]
        hit_at_k = float(np.mean(labeled_targets[top_indices] == 1))

    return {
        "average_precision": average_precision,
        "kendall_tau": tau,
        "spearman_rho": rho,
        "window_f1_at_positive_count": window_f1,
        "positive_hit_at_k": hit_at_k,
        "top_k": min(top_k, len(labeled_targets)),
        "window_count": len(rows),
        "labeled_window_count": int(np.sum(labeled_mask)),
        "positive_window_count": positive_count,
    }


def _measure_scores(
    scorer: Callable[[], np.ndarray],
    *,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, float | int]]:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    tracemalloc.start()
    started = time.perf_counter()
    scores = np.asarray(scorer(), dtype=np.float32).reshape(-1)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    _, python_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return scores, {
        "latency_seconds": elapsed,
        "python_peak_memory_bytes": int(python_peak),
        "cuda_peak_memory_bytes": (int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0),
    }


def _method_result(
    name: str,
    examples: list[WindowExample],
    scores: np.ndarray,
    timing: dict[str, float | int],
    *,
    top_k: int,
) -> dict[str, Any]:
    by_video: dict[str, list[int]] = defaultdict(list)
    for index, example in enumerate(examples):
        by_video[example.video_id].append(index)
    per_video = []
    for video_id, indices in sorted(by_video.items()):
        per_video.append(
            {
                "video_id": video_id,
                **window_metrics(
                    [examples[index] for index in indices],
                    scores[indices],
                    top_k=top_k,
                ),
            }
        )
    return {
        "method": name,
        **window_metrics(examples, scores, top_k=top_k),
        **timing,
        "per_video": per_video,
    }


def evaluate_manifest(
    *,
    manifest: str | Path,
    cache_dir: str | Path,
    checkpoint: str | Path,
    split: str,
    device: str,
    profiles: Iterable[str],
    top_k: int,
) -> dict[str, Any]:
    target_device = torch.device(
        "cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device
    )
    records = load_training_manifest(manifest, split=split)
    examples = build_window_examples(cache_dir, records)
    feature_rows = np.stack([example.feature for example in examples]).astype(np.float32)
    model, metadata = AdditiveAttentionScorer.load_checkpoint(
        checkpoint,
        device=target_device,
        expected_in_features=len(FEATURE_CHANNELS),
    )

    def score_ltr() -> np.ndarray:
        tensor = torch.as_tensor(feature_rows, dtype=torch.float32, device=target_device)
        with torch.no_grad():
            return model(tensor).reshape(-1).detach().cpu().numpy()

    ltr_scores, ltr_timing = _measure_scores(score_ltr, device=target_device)
    nms_per_video = []
    for record in records:
        video_id = str(record["video_id"])
        video_examples = [example for example in examples if example.video_id == video_id]
        indices = [index for index, example in enumerate(examples) if example.video_id == video_id]
        ordered = sorted(zip(video_examples, indices), key=lambda item: item[0].window_index)
        video_scores = np.asarray([ltr_scores[index] for _, index in ordered], dtype=np.float32)
        matrix = load_feature_matrix(cache_dir, video_id)
        timeline_scores = blend_scores(video_scores, T=matrix.shape[1])
        candidates = extract_topk_nms(
            timeline_scores,
            k=top_k,
            reference_duration=float(metadata["L_ref"]),
        )
        nms_per_video.append(
            {
                "video_id": video_id,
                "requested": top_k,
                "produced": len(candidates),
                "enough_for_render": len(candidates) >= 3,
                "candidate_ids": [candidate.candidate_id for candidate in candidates],
                "relaxation_stages": [
                    int(candidate.signals.get("relaxation_stage", 0))
                    for candidate in candidates
                ],
            }
        )
    methods = [
        _method_result("ltr", examples, ltr_scores, ltr_timing, top_k=top_k),
    ]
    for profile in profiles:
        profile_scores, profile_timing = _measure_scores(
            lambda selected=profile: profile_weight_scores(feature_rows, selected),
            device=torch.device("cpu"),
        )
        methods.append(
            _method_result(
                f"profile_weights:{profile}",
                examples,
                profile_scores,
                profile_timing,
                top_k=top_k,
            )
        )

    return {
        "schema_version": "1.0",
        "manifest": str(manifest),
        "cache_dir": str(cache_dir),
        "checkpoint": str(checkpoint),
        "split": split,
        "device": str(target_device),
        "video_count": len({record["video_id"] for record in records}),
        "window_count": len(examples),
        "dataset_fingerprint": metadata.get("dataset_fingerprint"),
        "validation_fingerprint": metadata.get("validation_fingerprint"),
        "checkpoint_epoch": metadata.get("epoch"),
        "checkpoint_val_ap": metadata.get("val_ap"),
        "nms_candidate_coverage": {
            "requested_per_video": top_k,
            "videos_with_render_minimum": sum(
                item["enough_for_render"] for item in nms_per_video
            ),
            "failure_rate": float(
                np.mean([not item["enough_for_render"] for item in nms_per_video])
            ),
            "per_video": nms_per_video,
        },
        "methods": methods,
        "metric_notes": {
            "average_precision": "Binary AP over positive/negative windows; ignored windows excluded.",
            "ranking": "Correlation against continuous window annotation scores.",
            "window_f1_at_positive_count": "Top-N window F1 where N equals the number of positives.",
            "positive_hit_at_k": "Fraction of top-K labeled windows classified positive.",
            "fscore_scope": "Window-level diagnostic, not TVSum/SumMe shot-level summary F-score.",
        },
    }


def _write_json(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "method",
        "video_id",
        "average_precision",
        "kendall_tau",
        "spearman_rho",
        "window_f1_at_positive_count",
        "positive_hit_at_k",
        "window_count",
        "labeled_window_count",
        "positive_window_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method in result["methods"]:
            for row in method["per_video"]:
                writer.writerow({"method": method["method"], **{field: row.get(field) for field in fields[1:]}})


def _write_markdown(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# LTR vs profile-weight evaluation",
        "",
        f"Split: `{result['split']}`; device: `{result['device']}`; "
        f"videos: {result['video_count']}; windows: {result['window_count']}.",
        "",
        "| Method | AP | Kendall tau | Spearman rho | Window F1 | Hit@K | Latency (s) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in result["methods"]:
        lines.append(
            f"| {method['method']} | {method['average_precision']:.6f} | "
            f"{method['kendall_tau']:.6f} | {method['spearman_rho']:.6f} | "
            f"{method['window_f1_at_positive_count']:.6f} | "
            f"{method['positive_hit_at_k']:.6f} | {method['latency_seconds']:.6f} |"
        )
    lines.extend(
        [
            "",
            "> F1 and Hit@K are window-level diagnostics. They are not the TVSum/SumMe shot-level F-score.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/manifests/tvsum_smoke.jsonl")
    parser.add_argument("--cache-dir", default="data/features_cache")
    parser.add_argument("--checkpoint", default="data/models/ltr_scorer.pt")
    parser.add_argument("--split", default="val")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--profiles", nargs="+", choices=PROFILE_NAMES, default=list(PROFILE_NAMES))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-json", default="data/reports/ltr_evaluation.json")
    parser.add_argument("--output-csv", default="data/reports/ltr_evaluation_per_video.csv")
    parser.add_argument("--output-markdown", default="data/reports/ltr_evaluation.md")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    result = evaluate_manifest(
        manifest=args.manifest,
        cache_dir=args.cache_dir,
        checkpoint=args.checkpoint,
        split=args.split,
        device=args.device,
        profiles=args.profiles,
        top_k=args.top_k,
    )
    _write_json(Path(args.output_json), result)
    _write_csv(Path(args.output_csv), result)
    _write_markdown(Path(args.output_markdown), result)
    print(json.dumps({key: value for key, value in result.items() if key != "methods"}, indent=2))
    for method in result["methods"]:
        print(
            f"{method['method']}: AP={method['average_precision']:.6f}, "
            f"tau={method['kendall_tau']:.6f}, rho={method['spearman_rho']:.6f}, "
            f"F1={method['window_f1_at_positive_count']:.6f}, "
            f"Hit@K={method['positive_hit_at_k']:.6f}"
        )
    return result


if __name__ == "__main__":
    main()
