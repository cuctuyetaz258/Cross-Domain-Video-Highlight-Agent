"""Resumable ActionFormer backbone pretraining on TVSum/SumMe."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from highlight_agent.ltr_contract import LTR_CHANNEL_ORDER, LTR_FEATURE_SCHEMA_VERSION
from highlight_agent.models.actionformer import (
    ActionFormerConfig,
    ActionFormerHighlightModel,
    save_actionformer_checkpoint,
)
from highlight_agent.models.training_artifacts import write_training_history_csv


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _records(path: Path, split: str) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line).get("split") == split
    ]


def _fingerprint(rows: list[dict[str, Any]]) -> str:
    payload = [{key: row.get(key) for key in ("video_id", "source", "split", "fps", "duration")} for row in rows]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _source_ranges(rows: list[dict[str, Any]]) -> dict[str, tuple[float, float]]:
    result: dict[str, tuple[float, float]] = {}
    for source in sorted({str(row["source"]) for row in rows}):
        values = np.concatenate(
            [np.asarray(row["frame_scores"], dtype=np.float32) for row in rows if row["source"] == source]
        )
        result[source] = (float(values.min()), float(values.max()))
    return result


def _timeline_labels(row: dict[str, Any], feature_length: int, sample_rate: float) -> np.ndarray:
    """Map source frame scores by timestamps, never by resizing a whole sequence."""
    fps, scores = float(row["fps"]), np.asarray(row["frame_scores"], dtype=np.float32)
    if fps <= 0 or not scores.size:
        raise ValueError(f"invalid benchmark labels for {row['video_id']}")
    source_time = np.arange(scores.size, dtype=np.float64) / fps
    cache_time = np.arange(feature_length, dtype=np.float64) / sample_rate
    return np.interp(cache_time, source_time, scores, left=float(scores[0]), right=float(scores[-1])).astype(np.float32)


def _level_zero_labels(
    row: dict[str, Any], matrix: np.ndarray, ranges: dict[str, tuple[float, float]], config: ActionFormerConfig
) -> np.ndarray:
    values = _timeline_labels(row, matrix.shape[1], config.input_sample_rate)
    low, high = ranges[str(row["source"])]
    values = (values - low) / max(high - low, 1e-6)
    usable = (values.size // config.downsample_factor) * config.downsample_factor
    if not usable:
        raise ValueError(f"feature cache is too short for {row['video_id']}")
    # Conv stem has kernel=stride=downsample_factor, so this is the exact support.
    return values[:usable].reshape(-1, config.downsample_factor).mean(axis=1).astype(np.float32)


def _example(
    row: dict[str, Any],
    cache_dir: Path,
    ranges: dict[str, tuple[float, float]],
    config: ActionFormerConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    matrix = np.load(cache_dir / str(row["video_id"]) / "feature_matrix.npy", allow_pickle=False)
    if matrix.dtype != np.float32 or matrix.ndim != 2 or matrix.shape[0] != config.in_features:
        raise ValueError(f"invalid cache for {row['video_id']}: {matrix.shape}/{matrix.dtype}")
    labels = _level_zero_labels(row, matrix, ranges, config)
    return torch.from_numpy(matrix).unsqueeze(0).to(device), torch.from_numpy(labels).reshape(1, 1, -1).to(device)


def _loss(prediction: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    if prediction.shape != labels.shape:
        raise ValueError(f"prediction/label timeline mismatch: {prediction.shape} != {labels.shape}")
    mse = functional.mse_loss(torch.sigmoid(prediction), labels)
    pieces: list[torch.Tensor] = []
    for offset in (1, 2, 4, 8, 16, 32):
        if prediction.shape[-1] <= offset:
            continue
        delta = labels[..., offset:] - labels[..., :-offset]
        valid = delta.abs() > 1e-3
        if valid.any():
            pieces.append(
                functional.softplus(-(prediction[..., offset:] - prediction[..., :-offset]) * delta.sign())[
                    valid
                ].mean()
            )
    pairwise = torch.stack(pieces).mean() if pieces else mse.new_zeros(())
    return mse + 0.2 * pairwise, {"mse": float(mse.detach().cpu()), "pairwise": float(pairwise.detach().cpu())}


def _ndcg(prediction: torch.Tensor, labels: torch.Tensor, k: int = 10) -> float:
    k = min(k, prediction.numel())
    gain = 2 ** labels.flatten() - 1
    discounts = 1 / torch.log2(torch.arange(k, dtype=torch.float32, device=prediction.device) + 2)
    actual = torch.sum(gain[torch.argsort(prediction.flatten(), descending=True)[:k]] * discounts)
    ideal = torch.sum(gain[torch.argsort(labels.flatten(), descending=True)[:k]] * discounts).clamp_min(1e-8)
    return float((actual / ideal).detach().cpu())


def _evaluate(
    model: ActionFormerHighlightModel,
    head: nn.Module,
    rows: list[dict[str, Any]],
    cache_dir: Path,
    ranges: dict[str, tuple[float, float]],
    config: ActionFormerConfig,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    head.eval()
    values: list[dict[str, float]] = []
    with torch.no_grad():
        for row in rows:
            features, labels = _example(row, cache_dir, ranges, config, device)
            prediction = head(model.backbone(features)[0][0])
            loss, metrics = _loss(prediction, labels)
            values.append({"loss": float(loss.cpu()), **metrics, "ndcg_at_10": _ndcg(prediction, labels)})
    return {key: float(np.mean([row[key] for row in values])) for key in values[0]}


def _preflight(rows: list[dict[str, Any]], cache_dir: Path, config: ActionFormerConfig) -> dict[str, Any]:
    if not rows:
        raise ValueError("benchmark manifest split is empty")
    checked = []
    for row in rows:
        path = cache_dir / str(row["video_id"]) / "feature_matrix.npy"
        matrix = np.load(path, mmap_mode="r", allow_pickle=False)
        if (
            matrix.dtype != np.float32
            or matrix.ndim != 2
            or matrix.shape[0] != config.in_features
            or matrix.shape[1] < config.downsample_factor
        ):
            raise ValueError(f"invalid cache for {row['video_id']}: {matrix.shape}/{matrix.dtype}")
        if not row.get("frame_scores"):
            raise ValueError(f"missing labels for {row['video_id']}")
        checked.append(str(row["video_id"]))
    return {"status": "passed", "checked_videos": checked, "feature_schema_version": LTR_FEATURE_SCHEMA_VERSION}


def _state(
    epoch: int, optimizer: AdamW, scheduler: CosineAnnealingLR, head: nn.Module, best: float, stale: int
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "importance_head_state_dict": head.state_dict(),
        "best_val_loss": best,
        "stale_epochs": stale,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _resume(
    path: Path, model: nn.Module, head: nn.Module, optimizer: AdamW, scheduler: CosineAnnealingLR, device: torch.device
) -> tuple[int, float, int]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state = checkpoint.get("training_state")
    if not isinstance(state, dict) or "importance_head_state_dict" not in state:
        raise ValueError("--resume needs a pretraining last.pt checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    head.load_state_dict(state["importance_head_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    scheduler.load_state_dict(state["scheduler_state_dict"])
    torch.set_rng_state(state["torch_rng_state"])
    if torch.cuda.is_available() and state.get("cuda_rng_state_all") is not None:
        torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])
    return int(state["epoch"]) + 1, float(state["best_val_loss"]), int(state["stale_epochs"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/manifests/tvsum_summe.jsonl")
    parser.add_argument("--cache-dir", default="data/features_cache")
    parser.add_argument("--run-dir", default="runs/actionformer/benchmark_pretrain")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-videos", type=int, default=None)
    parser.add_argument("--max-val-videos", type=int, default=None)
    args = parser.parse_args()
    if args.epochs <= 0 or args.patience <= 0:
        raise ValueError("epochs and patience must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device, config = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu")), ActionFormerConfig()
    manifest, cache_dir, run_dir = Path(args.manifest), Path(args.cache_dir), Path(args.run_dir)
    train, validation, test = (_records(manifest, split) for split in ("train", "val", "test"))
    if args.max_train_videos is not None:
        train = train[: args.max_train_videos]
    if args.max_val_videos is not None:
        validation = validation[: args.max_val_videos]
    ids = {
        name: [str(row["video_id"]) for row in rows]
        for name, rows in (("train", train), ("val", validation), ("test", test))
    }
    if (
        not train
        or not validation
        or any(set(ids[a]) & set(ids[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test")))
    ):
        raise ValueError("invalid or overlapping benchmark splits")
    preflight = _preflight(train + validation + test, cache_dir, config)
    run_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(run_dir / "preflight.json", preflight)
    ranges = _source_ranges(train)
    model = ActionFormerHighlightModel(config).to(device)
    head = nn.Conv1d(config.d_model, 1, 1).to(device)
    optimizer = AdamW(list(model.backbone.parameters()) + list(head.parameters()), lr=args.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    start, best, stale = (
        (1, float("inf"), 0)
        if not args.resume
        else _resume(Path(args.resume), model, head, optimizer, scheduler, device)
    )
    history: list[dict[str, Any]] = []
    best_path, last_path, report_path = run_dir / "best.pt", run_dir / "last.pt", run_dir / "run_report.json"
    metadata = {
        "feature_schema_version": LTR_FEATURE_SCHEMA_VERSION,
        "channel_order": list(LTR_CHANNEL_ORDER),
        "dataset_fingerprint": _fingerprint(train),
        "split_fingerprint": _fingerprint(train + validation + test),
        "normalization_policy_version": "benchmark_source_minmax_v2_timestamp_aligned",
        "pretraining": {
            "task": "frame_importance_ranking",
            "source_ranges": ranges,
            "label_alignment": "frame_timestamp_to_10hz_then_exact_stride_mean",
        },
        "data_lineage": {"train_video_ids": ids["train"], "selection_video_ids": ids["val"], "ancestors": []},
    }
    started = time.time()
    report: dict[str, Any] = {}
    try:
        for epoch in range(start, args.epochs + 1):
            model.train()
            head.train()
            train_metrics = []
            for row in random.Random(args.seed + epoch).sample(train, len(train)):
                features, labels = _example(row, cache_dir, ranges, config, device)
                optimizer.zero_grad()
                prediction = head(model.backbone(features)[0][0])
                loss, metrics = _loss(prediction, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(model.backbone.parameters()) + list(head.parameters()), 1.0)
                optimizer.step()
                train_metrics.append({"loss": float(loss.detach().cpu()), **metrics})
            val = _evaluate(model, head, validation, cache_dir, ranges, config, device)
            scheduler.step()
            improved = val["loss"] < best
            best, stale = (val["loss"], 0) if improved else (best, stale + 1)
            epoch_log = {
                "epoch": epoch,
                "train_loss": float(np.mean([x["loss"] for x in train_metrics])),
                "train_mse": float(np.mean([x["mse"] for x in train_metrics])),
                "train_pairwise": float(np.mean([x["pairwise"] for x in train_metrics])),
                "val_loss": val["loss"],
                "val_mse": val["mse"],
                "val_pairwise": val["pairwise"],
                "val_ndcg_at_10": val["ndcg_at_10"],
                "learning_rate": scheduler.get_last_lr()[0],
            }
            history.append(epoch_log)
            save_actionformer_checkpoint(
                last_path,
                model,
                metadata={**metadata, "checkpoint_role": "last", "epoch": epoch},
                training_state=_state(epoch, optimizer, scheduler, head, best, stale),
            )
            if improved:
                save_actionformer_checkpoint(
                    best_path, model, metadata={**metadata, "checkpoint_role": "best", "epoch": epoch}
                )
            report = {
                "status": "running",
                "device": device.type,
                "arguments": vars(args),
                "config": config.to_dict(),
                "preflight": preflight,
                "data_lineage": ids,
                "best_val_loss": best,
                "epochs": history,
                "artifacts": {
                    "best_checkpoint": str(best_path),
                    "last_checkpoint": str(last_path),
                    "history_csv": str(run_dir / "history.csv"),
                },
                "updated_at_unix": time.time(),
            }
            _atomic_json(report_path, report)
            write_training_history_csv(run_dir / "history.csv", history)
            if stale >= args.patience:
                break
    except BaseException as exc:
        _atomic_json(
            report_path,
            {
                "status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "artifacts": {"last_checkpoint": str(last_path)},
                "updated_at_unix": time.time(),
            },
        )
        raise
    report.update(status="complete", completed_at_unix=time.time(), elapsed_seconds=time.time() - started)
    _atomic_json(report_path, report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
