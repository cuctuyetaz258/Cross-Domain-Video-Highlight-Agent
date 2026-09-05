"""Pretrain an ActionFormer backbone on TVSum/SumMe importance timelines.

The benchmark has frame-level importance labels, not 30--90 second boundaries.
This trains only a temporary importance head and saves the compatible backbone
checkpoint for later custom localization fine-tuning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.optim import AdamW

from highlight_agent.features.ltr_contract import LTR_CHANNEL_ORDER, LTR_FEATURE_SCHEMA_VERSION
from highlight_agent.models.actionformer import ActionFormerConfig, ActionFormerHighlightModel, save_actionformer_checkpoint


def _records(path: Path, split: str) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [row for row in rows if row.get("split") == split]


def _source_ranges(records: list[dict]) -> dict[str, tuple[float, float]]:
    ranges: dict[str, tuple[float, float]] = {}
    for source in {str(row["source"]) for row in records}:
        values = np.concatenate([np.asarray(row["frame_scores"], dtype=np.float32) for row in records if row["source"] == source])
        ranges[source] = (float(values.min()), float(values.max()))
    return ranges


def _example(row: dict, cache_dir: Path, ranges: dict[str, tuple[float, float]], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    matrix = np.load(cache_dir / str(row["video_id"]) / "feature_matrix.npy", allow_pickle=False)
    if matrix.dtype != np.float32 or matrix.ndim != 2 or matrix.shape[0] != 7:
        raise ValueError(f"invalid cache for {row['video_id']}: {matrix.shape}/{matrix.dtype}")
    scores = np.asarray(row["frame_scores"], dtype=np.float32)
    lower, upper = ranges[str(row["source"])]
    target = (scores - lower) / max(upper - lower, 1e-6)
    features = torch.from_numpy(matrix).unsqueeze(0).to(device)
    labels = torch.from_numpy(target).reshape(1, 1, -1).to(device)
    return features, labels


def _loss(prediction: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    resized = functional.interpolate(labels, size=prediction.shape[-1], mode="linear", align_corners=False)
    regression = functional.mse_loss(torch.sigmoid(prediction), resized)
    # Rank pairs at a fixed offset so a video contributes independently of length.
    if prediction.shape[-1] < 2:
        return regression
    margin = resized[..., 1:] - resized[..., :-1]
    valid = margin.abs() > 1e-3
    pairwise = functional.softplus(-(prediction[..., 1:] - prediction[..., :-1]) * margin.sign())
    return regression + 0.2 * (pairwise[valid].mean() if valid.any() else pairwise.new_zeros(()))


def _evaluate(model: ActionFormerHighlightModel, head: nn.Module, rows: list[dict], cache_dir: Path, ranges: dict[str, tuple[float, float]], device: torch.device) -> float:
    model.eval(); head.eval(); losses = []
    with torch.no_grad():
        for row in rows:
            features, labels = _example(row, cache_dir, ranges, device)
            prediction = head(model.backbone(features)[0][0])
            losses.append(float(_loss(prediction, labels).cpu()))
    return float(np.mean(losses))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/manifests/tvsum_summe.jsonl")
    parser.add_argument("--cache-dir", default="data/features_cache")
    parser.add_argument("--output", default="data/models/actionformer_benchmark_pretrained.pt")
    parser.add_argument("--last-output", default="data/models/actionformer_benchmark_pretrained_last.pt")
    parser.add_argument("--report", default="data/reports/actionformer_benchmark_pretraining.json")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    train, validation = _records(Path(args.manifest), "train"), _records(Path(args.manifest), "val")
    if not train or not validation:
        raise ValueError("benchmark manifest must contain non-empty train and val splits")
    ranges = _source_ranges(train)
    config = ActionFormerConfig()
    model = ActionFormerHighlightModel(config).to(device)
    head = nn.Conv1d(config.d_model, 1, kernel_size=1).to(device)
    optimizer = AdamW(list(model.backbone.parameters()) + list(head.parameters()), lr=args.learning_rate)
    best, stale, history = float("inf"), 0, []
    source_ids = {split: [str(row["video_id"]) for row in rows] for split, rows in (("train", train), ("val", validation), ("test", _records(Path(args.manifest), "test")))}
    for epoch in range(1, args.epochs + 1):
        model.train(); head.train(); losses = []
        for row in random.Random(args.seed + epoch).sample(train, len(train)):
            features, labels = _example(row, Path(args.cache_dir), ranges, device)
            optimizer.zero_grad(); prediction = head(model.backbone(features)[0][0]); loss = _loss(prediction, labels)
            loss.backward(); torch.nn.utils.clip_grad_norm_(list(model.backbone.parameters()) + list(head.parameters()), 1.0); optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_loss = _evaluate(model, head, validation, Path(args.cache_dir), ranges, device)
        metadata = {"feature_schema_version": LTR_FEATURE_SCHEMA_VERSION, "channel_order": list(LTR_CHANNEL_ORDER), "dataset_fingerprint": hashlib.sha256(json.dumps(source_ids, sort_keys=True).encode()).hexdigest(), "split_fingerprint": hashlib.sha256(json.dumps(source_ids, sort_keys=True).encode()).hexdigest(), "normalization_policy_version": "benchmark_source_minmax_v1", "checkpoint_role": "last", "pretraining": {"task": "frame_importance_ranking", "source_ranges": ranges}, "data_lineage": {"train_video_ids": source_ids["train"], "selection_video_ids": source_ids["val"], "ancestors": []}}
        save_actionformer_checkpoint(args.last_output, model, metadata=metadata)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": val_loss})
        if val_loss < best:
            best, stale = val_loss, 0; save_actionformer_checkpoint(args.output, model, metadata={**metadata, "checkpoint_role": "best"})
        else: stale += 1
        if stale >= args.patience: break
    report = {"status": "complete", "device": device.type, "best_val_loss": best, "epochs": history, "artifacts": {"best_checkpoint": args.output, "last_checkpoint": args.last_output}, "data_lineage": source_ids, "completed_at_unix": time.time()}
    destination = Path(args.report); destination.parent.mkdir(parents=True, exist_ok=True); destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
