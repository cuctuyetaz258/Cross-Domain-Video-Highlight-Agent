from __future__ import annotations

import csv
import hashlib
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from highlight_agent.features.ltr_contract import LTR_CHANNEL_ORDER, LTR_FEATURE_SCHEMA_VERSION
from highlight_agent.models.proposal_ltr import (
    ProposalLTRConfig,
    build_proposal_ltr,
    pairwise_proposal_loss,
)
from highlight_agent.models.proposal_ltr_losses import ranknet_proposal_loss
from highlight_agent.models.training_artifacts import (
    write_training_curves_svg,
    write_training_history_csv,
)

from .actionformer import (
    ActionFormerConfig,
    ActionFormerHighlightModel,
    TemporalProposal,
    actionformer_losses,
    decode_proposals,
    load_actionformer_checkpoint,
    save_actionformer_checkpoint,
    soft_nms,
    temporal_iou,
)


@dataclass(frozen=True)
class ActionFormerExample:
    video_id: str
    domain: str
    duration: float
    features: np.ndarray
    boundaries: np.ndarray
    importance: np.ndarray


def load_actionformer_manifest(
    path: str | Path,
    *,
    split: str,
    project_root: str | Path = ".",
    ready_only: bool = True,
) -> list[ActionFormerExample]:
    root = Path(project_root).resolve()
    examples: list[ActionFormerExample] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("split") != split:
                continue
            if ready_only and not record.get("artifact_ready", False):
                continue
            feature_path = root / record["feature_path"]
            matrix = np.load(feature_path, allow_pickle=False)
            if matrix.dtype != np.float32 or matrix.ndim != 2 or matrix.shape[0] != 7:
                raise ValueError(f"invalid feature matrix for {record['video_id']}: {matrix.shape}/{matrix.dtype}")
            feature_duration = matrix.shape[1] / 10.0
            boundaries = np.asarray(
                [
                    [float(item["start"]), min(float(item["end"]), feature_duration)]
                    for item in record["highlights"]
                    if min(float(item["end"]), feature_duration) - float(item["start"]) >= 30.0 - 1e-6
                ],
                dtype=np.float32,
            ).reshape(-1, 2)
            if boundaries.size == 0:
                raise ValueError(f"no valid 30-90 second boundaries for {record['video_id']}")
            importance_path = root / record["importance_path"]
            with importance_path.open(encoding="utf-8-sig", newline="") as importance_handle:
                importance = np.asarray(
                    [
                        [float(row["start_sec"]), float(row["end_sec"]), float(row["importance"])]
                        for row in csv.DictReader(importance_handle)
                    ],
                    dtype=np.float32,
                ).reshape(-1, 3)
            examples.append(
                ActionFormerExample(
                    video_id=str(record["video_id"]),
                    domain=str(record["domain"]),
                    duration=feature_duration,
                    features=matrix,
                    boundaries=boundaries,
                    importance=importance,
                )
            )
    return examples


def _fingerprint(examples: Iterable[ActionFormerExample]) -> str:
    payload = [
        {
            "video_id": item.video_id,
            "domain": item.domain,
            "duration": item.duration,
            "boundaries": item.boundaries.tolist(),
            "shape": list(item.features.shape),
            "importance_shape": list(item.importance.shape),
        }
        for item in sorted(examples, key=lambda value: value.video_id)
    ]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _to_device(example: ActionFormerExample, device: torch.device) -> tuple[torch.Tensor, list[torch.Tensor]]:
    features = torch.from_numpy(example.features).unsqueeze(0).to(device)
    boundaries = [torch.from_numpy(example.boundaries).to(device)]
    return features, boundaries


def _ground_truth_proposals(example: ActionFormerExample) -> list[TemporalProposal]:
    return [
        TemporalProposal(float(start), float(end), 1.0, -1, index)
        for index, (start, end) in enumerate(example.boundaries)
    ]


def proposal_recall(
    proposals: list[TemporalProposal],
    ground_truth: list[TemporalProposal],
    *,
    k: int,
    threshold: float,
) -> float:
    if not ground_truth:
        return 0.0
    selected = proposals[:k]
    matched = sum(
        any(temporal_iou(prediction, target) >= threshold for prediction in selected)
        for target in ground_truth
    )
    return matched / len(ground_truth)


def evaluate_localization(
    model: ActionFormerHighlightModel,
    examples: list[ActionFormerExample],
    *,
    device: torch.device,
    lambda_reg: float,
    lambda_smooth: float,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    recalls: list[float] = []
    proposal_counts: list[int] = []
    with torch.no_grad():
        for example in examples:
            features, ground_truth = _to_device(example, device)
            outputs = model(features)
            loss = actionformer_losses(
                outputs,
                ground_truth,
                model.config,
                lambda_reg=lambda_reg,
                lambda_smooth=lambda_smooth,
            )
            proposals = soft_nms(
                decode_proposals(outputs, model.config, video_durations=[example.duration]),
                top_k=5,
            )
            losses.append(float(loss["total"].cpu()))
            proposal_counts.append(len(proposals))
            recalls.append(
                proposal_recall(
                    proposals,
                    _ground_truth_proposals(example),
                    k=3,
                    threshold=0.3,
                )
            )
    return {
        "total_loss": float(np.mean(losses)) if losses else float("nan"),
        "recall_at_3_iou_0_3": float(np.mean(recalls)) if recalls else 0.0,
        "mean_proposal_count": float(np.mean(proposal_counts)) if proposal_counts else 0.0,
    }


def train_actionformer_localization(
    *,
    train_examples: list[ActionFormerExample],
    val_examples: list[ActionFormerExample],
    output_path: str | Path,
    last_output_path: str | Path,
    log_path: str | Path,
    history_csv_path: str | Path,
    curves_path: str | Path,
    config: ActionFormerConfig,
    max_epochs: int = 30,
    learning_rate: float = 2e-4,
    weight_decay: float = 1e-4,
    lambda_reg: float = 1.0,
    lambda_smooth: float = 0.01,
    patience: int = 8,
    seed: int = 42,
    device: str | None = None,
    run_name: str = "actionformer_localization",
) -> tuple[ActionFormerHighlightModel, dict[str, Any]]:
    if not train_examples or not val_examples:
        raise ValueError("training and validation each require at least one ready video")
    if max_epochs <= 0 or patience <= 0:
        raise ValueError("max_epochs and patience must be positive")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    cuda_ready = torch.cuda.is_available() and torch.cuda.device_count() > 0
    target_device = torch.device(device or ("cuda" if cuda_ready else "cpu"))
    model = ActionFormerHighlightModel(config).to(target_device)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs)
    train_fingerprint = _fingerprint(train_examples)
    val_fingerprint = _fingerprint(val_examples)
    split_fingerprint = hashlib.sha256(f"{train_fingerprint}:{val_fingerprint}".encode()).hexdigest()
    history: list[dict[str, Any]] = []
    best_loss = float("inf")
    best_recall = -1.0
    best_epoch = 0
    stale_epochs = 0
    started = time.time()
    output = Path(output_path)
    last_output = Path(last_output_path)
    log = Path(log_path)
    history_csv = Path(history_csv_path)
    curves = Path(curves_path)
    base_metadata = {
        "feature_schema_version": LTR_FEATURE_SCHEMA_VERSION,
        "channel_order": list(LTR_CHANNEL_ORDER),
        "dataset_fingerprint": train_fingerprint,
        "validation_fingerprint": val_fingerprint,
        "split_fingerprint": split_fingerprint,
        "normalization_policy_version": "duration_30_90_v1",
        "run_name": run_name,
        "device": target_device.type,
        "seed": seed,
    }

    for epoch in range(1, max_epochs + 1):
        model.train()
        order = list(train_examples)
        random.Random(seed + epoch).shuffle(order)
        totals: dict[str, list[float]] = {"total": [], "focal": [], "regression": [], "smooth": []}
        positive_points = 0.0
        for example in order:
            features, ground_truth = _to_device(example, target_device)
            optimizer.zero_grad()
            losses = actionformer_losses(
                model(features),
                ground_truth,
                config,
                lambda_reg=lambda_reg,
                lambda_smooth=lambda_smooth,
            )
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            for name in totals:
                totals[name].append(float(losses[name].detach().cpu()))
            positive_points += float(losses["positive_points"].cpu())
        validation = evaluate_localization(
            model,
            val_examples,
            device=target_device,
            lambda_reg=lambda_reg,
            lambda_smooth=lambda_smooth,
        )
        scheduler.step()
        epoch_log = {
            "epoch": epoch,
            "train_total_loss": float(np.mean(totals["total"])),
            "train_focal_loss": float(np.mean(totals["focal"])),
            "train_regression_loss": float(np.mean(totals["regression"])),
            "train_smooth_loss": float(np.mean(totals["smooth"])),
            "train_positive_points": positive_points,
            "val_total_loss": validation["total_loss"],
            "val_recall_at_3_iou_0_3": validation["recall_at_3_iou_0_3"],
            "val_mean_proposal_count": validation["mean_proposal_count"],
            "selection_score": validation["recall_at_3_iou_0_3"],
            "selection_metric": "validation_recall_at_3_iou_0_3_then_loss",
            "learning_rate": scheduler.get_last_lr()[0],
        }
        history.append(epoch_log)
        checkpoint_metadata = {
            **base_metadata,
            "epoch": epoch,
            "selection_metric": epoch_log["selection_metric"],
            "selection_score": epoch_log["selection_score"],
            "validation_recall_at_3_iou_0_3": epoch_log["val_recall_at_3_iou_0_3"],
        }
        save_actionformer_checkpoint(last_output, model, metadata={**checkpoint_metadata, "checkpoint_role": "last"})
        recall_improved = validation["recall_at_3_iou_0_3"] > best_recall
        recall_tied = validation["recall_at_3_iou_0_3"] == best_recall
        if recall_improved or (recall_tied and validation["total_loss"] < best_loss):
            best_loss = validation["total_loss"]
            best_recall = validation["recall_at_3_iou_0_3"]
            best_epoch = epoch
            stale_epochs = 0
            save_actionformer_checkpoint(output, model, metadata={**checkpoint_metadata, "checkpoint_role": "best"})
        else:
            stale_epochs += 1
        report = {
            "schema_version": "1.0",
            "status": "running",
            "run_name": run_name,
            "started_at_unix": started,
            "updated_at_unix": time.time(),
            "config": config.to_dict(),
            "optimizer": {
                "name": "AdamW",
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "lambda_reg": lambda_reg,
                "lambda_smooth": lambda_smooth,
                "patience": patience,
                "max_epochs": max_epochs,
            },
            "data": {
                "train_video_ids": [item.video_id for item in train_examples],
                "val_video_ids": [item.video_id for item in val_examples],
                "train_fingerprint": train_fingerprint,
                "validation_fingerprint": val_fingerprint,
                "split_fingerprint": split_fingerprint,
            },
            "best_epoch": best_epoch,
            "best_val_total_loss": best_loss,
            "best_val_recall_at_3_iou_0_3": best_recall,
            "epochs": history,
            "artifacts": {
                "best_checkpoint": str(output),
                "last_checkpoint": str(last_output),
                "history_csv": str(history_csv),
                "curves_svg": str(curves),
            },
        }
        _atomic_json(log, report)
        write_training_history_csv(history_csv, history)
        write_training_curves_svg(
            curves,
            history,
            best_epoch=best_epoch,
            run_title=f"ActionFormer localization · {run_name}",
        )
        if stale_epochs >= patience:
            break

    report["status"] = "complete"
    report["completed_at_unix"] = time.time()
    report["elapsed_seconds"] = report["completed_at_unix"] - started
    _atomic_json(log, report)
    best_model, _, _ = load_actionformer_checkpoint(output, device="cpu")
    return best_model, report


def proposal_utility(
    example: ActionFormerExample,
    proposal: TemporalProposal,
    *,
    mean_weight: float = 0.45,
    top_weight: float = 0.25,
    tiou_weight: float = 0.30,
) -> float:
    """Continuous proposal-quality label combining content and boundary agreement."""

    if min(mean_weight, top_weight, tiou_weight) < 0:
        raise ValueError("utility weights must be non-negative")
    weight_total = mean_weight + top_weight + tiou_weight
    if weight_total <= 0:
        raise ValueError("at least one utility weight must be positive")
    overlaps: list[tuple[float, float]] = []
    for start, end, importance in example.importance:
        overlap = max(0.0, min(proposal.end, float(end)) - max(proposal.start, float(start)))
        if overlap > 0:
            overlaps.append((overlap, float(importance)))
    if overlaps:
        weighted_mean = sum(overlap * value for overlap, value in overlaps) / sum(
            overlap for overlap, _ in overlaps
        )
        top_count = max(1, int(np.ceil(len(overlaps) * 0.2)))
        top_mean = float(np.mean(sorted((value for _, value in overlaps), reverse=True)[:top_count]))
        normalized_mean = min(max((weighted_mean - 1.0) / 4.0, 0.0), 1.0)
        normalized_top = min(max((top_mean - 1.0) / 4.0, 0.0), 1.0)
    else:
        normalized_mean = 0.0
        normalized_top = 0.0
    targets = _ground_truth_proposals(example)
    max_tiou = max((temporal_iou(proposal, target) for target in targets), default=0.0)
    utility = (
        mean_weight * normalized_mean
        + top_weight * normalized_top
        + tiou_weight * max_tiou
    ) / weight_total
    return min(max(float(utility), 0.0), 1.0)


def proposal_training_set(
    example: ActionFormerExample,
    predicted_proposals: list[TemporalProposal] | None = None,
) -> list[TemporalProposal]:
    candidates: dict[tuple[float, float], TemporalProposal] = {}
    for proposal in predicted_proposals or []:
        candidates[(round(proposal.start, 3), round(proposal.end, 3))] = proposal
    for index, (start, end) in enumerate(example.boundaries):
        proposal = TemporalProposal(float(start), float(end), 1.0, -1, index)
        candidates[(round(proposal.start, 3), round(proposal.end, 3))] = proposal
    index = len(candidates)
    for duration in (30.0, 60.0, 90.0):
        for start in np.arange(0.0, max(0.0, example.duration - duration) + 1e-6, 30.0):
            proposal = TemporalProposal(float(start), float(start + duration), 0.5, -1, index)
            candidates[(round(proposal.start, 3), round(proposal.end, 3))] = proposal
            index += 1
    return sorted(candidates.values(), key=lambda item: (item.start, item.end))


def _evaluate_proposal_ltr(
    actionformer: ActionFormerHighlightModel,
    scorer: nn.Module,
    examples: list[ActionFormerExample],
    *,
    device: torch.device,
    margin: float,
    utility_delta: float,
    loss_type: str,
    pair_weighting: str,
    ndcg_k: int,
    gain_scale: float,
    max_pairs_per_video: int | None,
    predicted_proposals_by_video: dict[str, list[TemporalProposal]] | None = None,
) -> dict[str, float]:
    actionformer.eval()
    scorer.eval()
    losses: list[float] = []
    ndcgs: list[float] = []
    with torch.no_grad():
        for example in examples:
            features, _ = _to_device(example, device)
            outputs = actionformer(features)
            base_features = outputs["features"][0]
            predicted = _training_predictions(
                example,
                outputs,
                actionformer,
                predicted_proposals_by_video,
            )
            proposals = proposal_training_set(example, predicted)
            scores, _ = scorer(base_features, [proposals], stride_seconds=actionformer.config.base_stride_seconds)
            utilities = scores.new_tensor([proposal_utility(example, proposal) for proposal in proposals])
            video_indices = torch.zeros_like(scores, dtype=torch.long)
            loss = _proposal_pairwise_loss(
                scores,
                utilities,
                video_indices,
                loss_type=loss_type,
                margin=margin,
                utility_delta=utility_delta,
                pair_weighting=pair_weighting,
                ndcg_k=ndcg_k,
                gain_scale=gain_scale,
                max_pairs_per_video=max_pairs_per_video,
            )
            losses.append(float(loss.cpu()))
            k = min(3, scores.numel())
            predicted = torch.topk(scores, k).indices
            ideal = torch.topk(utilities, k).indices
            discounts = 1 / torch.log2(torch.arange(k, device=device, dtype=torch.float32) + 2)
            dcg = torch.sum((2 ** utilities[predicted] - 1) * discounts)
            idcg = torch.sum((2 ** utilities[ideal] - 1) * discounts).clamp_min(1e-8)
            ndcgs.append(float((dcg / idcg).cpu()))
    return {
        "pairwise_loss": float(np.mean(losses)) if losses else float("nan"),
        "ndcg_at_3": float(np.mean(ndcgs)) if ndcgs else 0.0,
    }


def _proposal_pairwise_loss(
    scores: torch.Tensor,
    utilities: torch.Tensor,
    video_indices: torch.Tensor,
    *,
    loss_type: str,
    margin: float,
    utility_delta: float,
    pair_weighting: str,
    ndcg_k: int,
    gain_scale: float,
    max_pairs_per_video: int | None,
) -> torch.Tensor:
    if loss_type == "margin":
        return pairwise_proposal_loss(
            scores,
            utilities,
            video_indices,
            margin=margin,
            utility_delta=utility_delta,
        )
    if loss_type == "ranknet":
        return ranknet_proposal_loss(
            scores,
            utilities,
            video_indices,
            utility_delta=utility_delta,
            weighting=pair_weighting,
            ndcg_k=ndcg_k,
            gain_scale=gain_scale,
            max_pairs_per_video=max_pairs_per_video,
        )
    raise ValueError(f"unsupported proposal pairwise loss: {loss_type}")


def _training_predictions(
    example: ActionFormerExample,
    outputs: dict[str, list[torch.Tensor]],
    actionformer: ActionFormerHighlightModel,
    predicted_proposals_by_video: dict[str, list[TemporalProposal]] | None,
) -> list[TemporalProposal]:
    if predicted_proposals_by_video is not None:
        if example.video_id not in predicted_proposals_by_video:
            raise ValueError(f"proposal cache is missing video {example.video_id}")
        return predicted_proposals_by_video[example.video_id]
    return decode_proposals(
        outputs,
        actionformer.config,
        video_durations=[example.duration],
    )


def train_proposal_ltr(
    *,
    actionformer: ActionFormerHighlightModel,
    checkpoint_metadata: dict[str, Any],
    train_examples: list[ActionFormerExample],
    val_examples: list[ActionFormerExample],
    output_path: str | Path,
    last_output_path: str | Path,
    log_path: str | Path,
    history_csv_path: str | Path,
    curves_path: str | Path,
    max_epochs: int = 30,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    margin: float = 1.0,
    utility_delta: float = 0.1,
    scorer_config: ProposalLTRConfig | None = None,
    loss_type: str = "ranknet",
    pair_weighting: str = "utility",
    ndcg_k: int = 3,
    gain_scale: float = 4.0,
    max_pairs_per_video: int | None = 256,
    patience: int = 8,
    seed: int = 42,
    device: str | None = None,
    run_name: str = "proposal_ltr",
    predicted_proposals_by_video: dict[str, list[TemporalProposal]] | None = None,
    proposal_cache_metadata: dict[str, Any] | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    if not train_examples or not val_examples:
        raise ValueError("training and validation each require at least one ready video")
    if max_epochs <= 0 or patience <= 0:
        raise ValueError("max_epochs and patience must be positive")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cuda_ready = torch.cuda.is_available() and torch.cuda.device_count() > 0
    target_device = torch.device(device or ("cuda" if cuda_ready else "cpu"))
    actionformer.to(target_device).eval()
    for parameter in actionformer.parameters():
        parameter.requires_grad_(False)
    resolved_scorer_config = scorer_config or ProposalLTRConfig()
    scorer = build_proposal_ltr(actionformer.config.d_model, resolved_scorer_config).to(target_device)
    optimizer = AdamW(scorer.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs)
    history: list[dict[str, Any]] = []
    best_ndcg = -1.0
    best_epoch = 0
    stale_epochs = 0
    started = time.time()
    output = Path(output_path)
    last_output = Path(last_output_path)
    log = Path(log_path)
    history_csv = Path(history_csv_path)
    curves = Path(curves_path)
    report: dict[str, Any] = {}
    for epoch in range(1, max_epochs + 1):
        scorer.train()
        order = list(train_examples)
        random.Random(seed + epoch).shuffle(order)
        epoch_losses: list[float] = []
        for example in order:
            features, _ = _to_device(example, target_device)
            with torch.no_grad():
                outputs = actionformer(features)
                base_features = outputs["features"][0]
                predicted = _training_predictions(
                    example,
                    outputs,
                    actionformer,
                    predicted_proposals_by_video,
                )
            proposals = proposal_training_set(example, predicted)
            scores, _ = scorer(
                base_features,
                [proposals],
                stride_seconds=actionformer.config.base_stride_seconds,
            )
            utilities = scores.new_tensor([proposal_utility(example, proposal) for proposal in proposals])
            video_indices = torch.zeros_like(scores, dtype=torch.long)
            optimizer.zero_grad()
            loss = _proposal_pairwise_loss(
                scores,
                utilities,
                video_indices,
                loss_type=loss_type,
                margin=margin,
                utility_delta=utility_delta,
                pair_weighting=pair_weighting,
                ndcg_k=ndcg_k,
                gain_scale=gain_scale,
                max_pairs_per_video=max_pairs_per_video,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(scorer.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        validation = _evaluate_proposal_ltr(
            actionformer,
            scorer,
            val_examples,
            device=target_device,
            margin=margin,
            utility_delta=utility_delta,
            loss_type=loss_type,
            pair_weighting=pair_weighting,
            ndcg_k=ndcg_k,
            gain_scale=gain_scale,
            max_pairs_per_video=max_pairs_per_video,
            predicted_proposals_by_video=predicted_proposals_by_video,
        )
        scheduler.step()
        epoch_log = {
            "epoch": epoch,
            "train_pairwise_loss": float(np.mean(epoch_losses)),
            "train_total_loss": float(np.mean(epoch_losses)),
            "val_pairwise_loss": validation["pairwise_loss"],
            "val_ndcg_at_3": validation["ndcg_at_3"],
            "selection_score": validation["ndcg_at_3"],
            "selection_metric": "validation_ndcg_at_3",
            "learning_rate": scheduler.get_last_lr()[0],
        }
        history.append(epoch_log)
        metadata = {
            **checkpoint_metadata,
            "run_name": run_name,
            "device": target_device.type,
            "training_stage": "proposal_ltr",
            "proposal_ltr_config": {
                **resolved_scorer_config.to_dict(),
                "channels": actionformer.config.d_model,
                "loss_type": loss_type,
                "pair_weighting": pair_weighting,
                "margin": margin,
                "utility_delta": utility_delta,
                "ndcg_k": ndcg_k,
                "gain_scale": gain_scale,
                "max_pairs_per_video": max_pairs_per_video,
                "utility_version": "proposal_utility_v2",
                "utility_weights": {
                    "coverage_mean": 0.45,
                    "top_20_percent": 0.25,
                    "max_tiou": 0.30,
                },
            },
            "proposal_ltr_epoch": epoch,
            "proposal_ltr_selection_metric": epoch_log["selection_metric"],
            "proposal_ltr_selection_score": epoch_log["selection_score"],
        }
        save_actionformer_checkpoint(
            last_output,
            actionformer,
            metadata={**metadata, "checkpoint_role": "last"},
            proposal_ltr_state_dict=scorer.state_dict(),
        )
        if validation["ndcg_at_3"] > best_ndcg:
            best_ndcg = validation["ndcg_at_3"]
            best_epoch = epoch
            stale_epochs = 0
            save_actionformer_checkpoint(
                output,
                actionformer,
                metadata={**metadata, "checkpoint_role": "best"},
                proposal_ltr_state_dict=scorer.state_dict(),
            )
        else:
            stale_epochs += 1
        report = {
            "schema_version": "1.0",
            "status": "running",
            "run_name": run_name,
            "stage": "proposal_ltr",
            "started_at_unix": started,
            "updated_at_unix": time.time(),
            "config": {
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "margin": margin,
                "utility_delta": utility_delta,
                "loss_type": loss_type,
                "pair_weighting": pair_weighting,
                "ndcg_k": ndcg_k,
                "gain_scale": gain_scale,
                "max_pairs_per_video": max_pairs_per_video,
                "scorer": resolved_scorer_config.to_dict(),
                "patience": patience,
                "max_epochs": max_epochs,
                "device": target_device.type,
                "seed": seed,
            },
            "data": {
                "train_video_ids": [item.video_id for item in train_examples],
                "val_video_ids": [item.video_id for item in val_examples],
                "proposal_source": "oof_cache" if predicted_proposals_by_video is not None else "online",
                "proposal_cache": proposal_cache_metadata,
            },
            "best_epoch": best_epoch,
            "best_val_ndcg_at_3": best_ndcg,
            "epochs": history,
            "artifacts": {
                "best_checkpoint": str(output),
                "last_checkpoint": str(last_output),
                "history_csv": str(history_csv),
                "curves_svg": str(curves),
            },
        }
        _atomic_json(log, report)
        write_training_history_csv(history_csv, history)
        write_training_curves_svg(
            curves,
            history,
            best_epoch=best_epoch,
            run_title=f"Proposal LTR · {run_name}",
        )
        if stale_epochs >= patience:
            break
    report["status"] = "complete"
    report["completed_at_unix"] = time.time()
    report["elapsed_seconds"] = report["completed_at_unix"] - started
    _atomic_json(log, report)
    _, best_metadata, best_state = load_actionformer_checkpoint(output, device="cpu")
    best_scorer = build_proposal_ltr(actionformer.config.d_model, resolved_scorer_config)
    if best_state is None:
        raise ValueError("best proposal LTR checkpoint did not contain scorer weights")
    best_scorer.load_state_dict(best_state)
    best_scorer.eval()
    return best_scorer, {**report, "checkpoint_metadata": best_metadata}
