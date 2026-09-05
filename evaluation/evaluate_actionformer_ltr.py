"""Evaluate an ActionFormer-LTR checkpoint on one manifest split."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from highlight_agent.models.actionformer import (
    TemporalProposal,
    decode_proposals,
    load_actionformer_checkpoint,
    soft_nms,
    temporal_iou,
)
from highlight_agent.models.proposal_ltr import ProposalLTRConfig, build_proposal_ltr
from highlight_agent.models.train_actionformer_ltr import (
    ActionFormerExample,
    load_actionformer_manifest,
)


def detection_average_precision(
    predictions: list[tuple[str, TemporalProposal]],
    ground_truths: dict[str, list[TemporalProposal]],
    *,
    iou_threshold: float,
) -> float:
    """Compute dataset-level interpolated AP with greedy per-video matching."""

    total_ground_truth = sum(len(items) for items in ground_truths.values())
    if total_ground_truth == 0:
        return 0.0
    matched = {video_id: set() for video_id in ground_truths}
    true_positives: list[float] = []
    false_positives: list[float] = []
    for video_id, prediction in sorted(
        predictions,
        key=lambda item: (-item[1].score, item[0], item[1].start),
    ):
        targets = ground_truths.get(video_id, [])
        available = [
            (index, temporal_iou(prediction, target))
            for index, target in enumerate(targets)
            if index not in matched.setdefault(video_id, set())
        ]
        best_index, best_iou = max(available, key=lambda item: item[1], default=(-1, 0.0))
        is_match = best_index >= 0 and best_iou >= iou_threshold
        true_positives.append(float(is_match))
        false_positives.append(float(not is_match))
        if is_match:
            matched[video_id].add(best_index)
    if not true_positives:
        return 0.0
    tp = np.cumsum(true_positives)
    fp = np.cumsum(false_positives)
    recall = tp / total_ground_truth
    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    changed = np.where(recall[1:] != recall[:-1])[0]
    return float(np.sum((recall[changed + 1] - recall[changed]) * precision[changed + 1]))


def _targets(example: ActionFormerExample) -> list[TemporalProposal]:
    return [
        TemporalProposal(float(start), float(end), 1.0, -1, index)
        for index, (start, end) in enumerate(example.boundaries)
    ]


def _recall_at_k(
    predictions: list[TemporalProposal],
    targets: list[TemporalProposal],
    *,
    k: int,
    threshold: float,
) -> float:
    if not targets:
        return 0.0
    return sum(
        any(temporal_iou(prediction, target) >= threshold for prediction in predictions[:k])
        for target in targets
    ) / len(targets)


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    examples: list[ActionFormerExample],
    *,
    device: str | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    if not examples:
        raise ValueError("evaluation requires at least one ready example")
    cuda_ready = torch.cuda.is_available() and torch.cuda.device_count() > 0
    target_device = torch.device(device or ("cuda" if cuda_ready else "cpu"))
    model, metadata, proposal_state = load_actionformer_checkpoint(
        checkpoint_path,
        device=target_device,
    )
    scorer = None
    if proposal_state is not None:
        scorer_payload = metadata.get("proposal_ltr_config")
        scorer_config = (
            ProposalLTRConfig.from_dict(scorer_payload)
            if isinstance(scorer_payload, dict)
            else ProposalLTRConfig(architecture="mlp", d_model=128, dropout=0.1)
        )
        scorer = build_proposal_ltr(model.config.d_model, scorer_config).to(target_device)
        scorer.load_state_dict(proposal_state)
        scorer.eval()
    per_video: list[dict[str, Any]] = []
    all_predictions: list[tuple[str, TemporalProposal]] = []
    all_targets: dict[str, list[TemporalProposal]] = {}
    started = time.perf_counter()
    with torch.no_grad():
        for example in examples:
            features = torch.from_numpy(example.features).unsqueeze(0).to(target_device)
            outputs = model(features)
            proposals = decode_proposals(
                outputs,
                model.config,
                video_durations=[example.duration],
            )
            if scorer is not None and proposals:
                scores, provenance = scorer(
                    outputs["features"][0],
                    [proposals],
                    stride_seconds=model.config.base_stride_seconds,
                )
                rank_probabilities = torch.sigmoid(scores).tolist()
                for score, (_, proposal_index) in zip(rank_probabilities, provenance):
                    proposals[proposal_index] = replace(
                        proposals[proposal_index],
                        rank_score=float(score),
                    )
            predictions = soft_nms(proposals, top_k=top_k)
            targets = _targets(example)
            all_targets[example.video_id] = targets
            all_predictions.extend((example.video_id, item) for item in predictions)
            best_ious = [
                max((temporal_iou(prediction, target) for target in targets), default=0.0)
                for prediction in predictions
            ]
            boundary_errors = []
            for target in targets:
                best = max(predictions, key=lambda item: temporal_iou(item, target), default=None)
                if best is not None:
                    boundary_errors.append(
                        (abs(best.start - target.start) + abs(best.end - target.end)) / 2
                    )
            per_video.append(
                {
                    "video_id": example.video_id,
                    "domain": example.domain,
                    "ground_truth_count": len(targets),
                    "proposal_count": len(predictions),
                    "recall_at_1_iou_0_3": _recall_at_k(
                        predictions, targets, k=1, threshold=0.3
                    ),
                    "recall_at_3_iou_0_3": _recall_at_k(
                        predictions, targets, k=3, threshold=0.3
                    ),
                    "recall_at_5_iou_0_3": _recall_at_k(
                        predictions, targets, k=5, threshold=0.3
                    ),
                    "mean_iou": float(np.mean(best_ious)) if best_ious else 0.0,
                    "mean_boundary_error_seconds": (
                        float(np.mean(boundary_errors)) if boundary_errors else None
                    ),
                }
            )
    elapsed = time.perf_counter() - started
    aps = {
        f"map_iou_{threshold:.1f}".replace(".", "_"): detection_average_precision(
            all_predictions,
            all_targets,
            iou_threshold=threshold,
        )
        for threshold in (0.3, 0.5, 0.7)
    }
    aggregate_keys = (
        "recall_at_1_iou_0_3",
        "recall_at_3_iou_0_3",
        "recall_at_5_iou_0_3",
        "mean_iou",
    )
    return {
        "schema_version": "1.0",
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "checkpoint_metadata": metadata,
        "device": target_device.type,
        "video_count": len(examples),
        "elapsed_seconds": elapsed,
        "seconds_per_video": elapsed / len(examples),
        "proposal_ltr_enabled": scorer is not None,
        "metrics": {
            **aps,
            **{
                key: float(np.mean([record[key] for record in per_video]))
                for key in aggregate_keys
            },
            "duration_valid_rate": float(
                np.mean(
                    [
                        model.config.min_duration_seconds
                        <= prediction.duration
                        <= model.config.max_duration_seconds
                        for _, prediction in all_predictions
                    ]
                )
            )
            if all_predictions
            else 0.0,
        },
        "per_video": per_video,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default="data/manifests/actionformer_fold0.jsonl")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output", default="data/reports/actionformer_evaluation.json")
    parser.add_argument("--device", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    examples = load_actionformer_manifest(
        args.manifest,
        split=args.split,
        project_root=args.project_root,
    )
    report = evaluate_checkpoint(
        args.checkpoint,
        examples,
        device=args.device,
        top_k=args.top_k,
    )
    report.update({"manifest": args.manifest, "split": args.split})
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(destination)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
