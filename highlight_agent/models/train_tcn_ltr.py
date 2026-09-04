"""Train or fine-tune the V2 non-causal TCN Learning-to-Rank scorer."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from highlight_agent.features.ltr_contract import feature_contract

from .tcn_ltr_scorer import TemporalConvLTRScorer
from .train_offline import (
    FEATURE_SCHEMA_VERSION,
    WindowExample,
    _records_fingerprint,
    _set_seed,
    build_balanced_epoch_pairs,
    build_window_examples,
    compute_lref,
    load_training_manifest,
    margin_ranking_loss,
    temporal_smoothness_loss,
)
from .training_artifacts import write_training_curves_svg, write_training_history_csv


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sequences(examples: Iterable[WindowExample]) -> dict[str, list[WindowExample]]:
    grouped: dict[str, list[WindowExample]] = defaultdict(list)
    for example in examples:
        grouped[example.video_id].append(example)
    for rows in grouped.values():
        rows.sort(key=lambda row: row.window_index)
    return dict(grouped)


def score_examples(
    model: TemporalConvLTRScorer, examples: Iterable[WindowExample], *, device: torch.device
) -> dict[int, float]:
    """Score every window in its own chronological video sequence."""

    model.eval()
    scores: dict[int, float] = {}
    with torch.no_grad():
        for rows in _sequences(examples).values():
            features = torch.as_tensor(np.stack([row.feature for row in rows]), dtype=torch.float32, device=device)
            output = model(features).detach().cpu().numpy()
            scores.update({id(row): float(value) for row, value in zip(rows, output, strict=True)})
    return scores


def evaluate_macro_video_ap(
    model: TemporalConvLTRScorer, examples: Iterable[WindowExample], *, device: torch.device
) -> tuple[float, dict[str, float]]:
    rows = list(examples)
    predicted = score_examples(model, rows, device=device)
    by_video = _sequences(rows)
    per_video: dict[str, float] = {}
    for video_id, video_rows in by_video.items():
        labeled = [row for row in video_rows if row.label in {0, 1}]
        targets = np.asarray([row.label for row in labeled])
        if set(targets) == {0, 1}:
            per_video[video_id] = float(average_precision_score(targets, [predicted[id(row)] for row in labeled]))
    if not per_video:
        raise ValueError("selection data must contain a positive and negative window in at least one video")
    return float(np.mean(list(per_video.values()))), per_video


def train(
    *,
    cache_dir: str | Path,
    records: list[dict[str, Any]],
    val_records: list[dict[str, Any]] | None,
    output: str | Path,
    init_checkpoint: str | Path | None,
    hidden_dim: int = 32,
    dilations: tuple[int, ...] = (1, 2, 4, 8),
    dropout: float = 0.1,
    gamma: float = 1.0,
    lambda_smooth: float = 0.01,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    batch_size: int = 32,
    max_epochs: int = 50,
    patience: int = 15,
    seed: int = 42,
    max_pairs_per_video: int = 2048,
    training_log: str | Path | None = None,
    history_csv: str | Path | None = None,
    training_plot: str | Path | None = None,
    last_checkpoint: str | Path | None = None,
) -> TemporalConvLTRScorer:
    """Train V2 with within-video pair sampling and sequence-aware scoring."""

    if not records:
        raise ValueError("training records must not be empty")
    _set_seed(seed)
    target = Path(output)
    log_path = Path(training_log or target.with_name(f"{target.stem}_log.json"))
    history_path = Path(history_csv or log_path.with_name(f"{log_path.stem}_history.csv"))
    plot_path = Path(training_plot or log_path.with_name(f"{log_path.stem}_curves.svg"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_rows = build_window_examples(cache_dir, records)
    val_rows = build_window_examples(cache_dir, val_records) if val_records else []
    selection_rows = val_rows or train_rows
    selection_split = "validation" if val_rows else "training"
    pairs, initial_balance = build_balanced_epoch_pairs(
        train_rows, max_pairs_per_video=max_pairs_per_video, seed=seed
    )
    if not pairs:
        raise ValueError("training data must contain positive/negative pairs within a video")

    parent: dict[str, Any] | None = None
    if init_checkpoint:
        model, parent_metadata = TemporalConvLTRScorer.load_checkpoint(init_checkpoint, device=device)
        if model.hidden_dim != hidden_dim or model.dilations != dilations:
            raise ValueError("init checkpoint architecture does not match requested V2 architecture")
        parent = {"path": str(Path(init_checkpoint).resolve()), "metadata": parent_metadata}
    else:
        model = TemporalConvLTRScorer(hidden_dim=hidden_dim, dilations=dilations, dropout=dropout).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs)
    config = {
        "model_type": TemporalConvLTRScorer.model_type,
        "hidden_dim": hidden_dim,
        "dilations": list(dilations),
        "dropout": dropout,
        "receptive_field_tokens": model.receptive_field_tokens,
        "gamma": gamma,
        "lambda_smooth": lambda_smooth,
        "lr": lr,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "patience": patience,
        "seed": seed,
        "device": device.type,
        "max_pairs_per_video": max_pairs_per_video,
        "initial_epoch_pair_count": initial_balance["epoch_pair_count"],
    }
    best_ap, best_epoch, stale, history = -1.0, 0, 0, []
    chronological = _sequences(train_rows)
    for epoch in range(1, max_epochs + 1):
        model.train()
        pairs, balance = build_balanced_epoch_pairs(
            train_rows, max_pairs_per_video=max_pairs_per_video, seed=seed + epoch
        )
        margin_sum = 0.0
        for offset in range(0, len(pairs), batch_size):
            batch = pairs[offset : offset + batch_size]
            optimizer.zero_grad()
            pair_groups: dict[str, list[tuple[WindowExample, WindowExample]]] = defaultdict(list)
            for pair in batch:
                pair_groups[pair[0].video_id].append(pair)
            losses: list[torch.Tensor] = []
            # Forward each video once, retaining context while still batching ranking pairs.
            for video_id, video_pairs in pair_groups.items():
                video_rows = chronological[video_id]
                features = torch.as_tensor(np.stack([row.feature for row in video_rows]), dtype=torch.float32, device=device)
                sequence_scores = model(features)
                for positive, negative in video_pairs:
                    losses.append(
                        margin_ranking_loss(
                            sequence_scores[positive.window_index], sequence_scores[negative.window_index], gamma
                        )
                    )
            loss = torch.stack(losses).mean()
            loss.backward()
            optimizer.step()
            margin_sum += float(loss.detach().cpu()) * len(batch)

        smooth_values = []
        if lambda_smooth:
            for rows in chronological.values():
                features = torch.as_tensor(np.stack([row.feature for row in rows]), dtype=torch.float32, device=device)
                optimizer.zero_grad()
                smooth = temporal_smoothness_loss([model(features)])
                (lambda_smooth * smooth).backward()
                optimizer.step()
                smooth_values.append(float(smooth.detach().cpu()))
        scheduler.step()
        selection_ap, per_video = evaluate_macro_video_ap(model, selection_rows, device=device)
        smooth_mean = float(np.mean(smooth_values)) if smooth_values else 0.0
        row = {
            "epoch": epoch,
            "train_margin_loss": margin_sum / len(pairs),
            "train_smooth_loss": smooth_mean,
            "train_total_loss": margin_sum / len(pairs) + lambda_smooth * smooth_mean,
            "selection_ap": selection_ap,
            "selection_split": selection_split,
            "selection_metric": "macro_video_average_precision",
            "learning_rate": scheduler.get_last_lr()[0],
            "ap_by_video": per_video,
            "source_pair_counts": balance["source_pair_counts"],
        }
        history.append(row)
        if selection_ap > best_ap:
            best_ap, best_epoch, stale = selection_ap, epoch, 0
            model.save(target, metadata={
                "schema_version": FEATURE_SCHEMA_VERSION,
                "feature_schema": feature_contract(),
                "model_type": TemporalConvLTRScorer.model_type,
                "L_ref": compute_lref(records),
                "epoch": epoch,
                "selection_ap": selection_ap,
                "selection_split": selection_split,
                "dataset_fingerprint": _records_fingerprint(records),
                "validation_fingerprint": _records_fingerprint(val_records or []),
                "config": config,
                "parent_checkpoint": parent,
            })
        else:
            stale += 1
        if last_checkpoint:
            model.save(Path(last_checkpoint), metadata={
                "schema_version": FEATURE_SCHEMA_VERSION,
                "feature_schema": feature_contract(),
                "model_type": TemporalConvLTRScorer.model_type,
                "L_ref": compute_lref(records),
                "epoch": epoch,
                "selection_ap": selection_ap,
                "selection_split": selection_split,
                "dataset_fingerprint": _records_fingerprint(records),
                "validation_fingerprint": _records_fingerprint(val_records or []),
                "config": config,
                "checkpoint_role": "last",
                "parent_checkpoint": parent,
            })
        _write_json(log_path, {"config": config, "best_epoch": best_epoch, "best_ap": best_ap, "epochs": history})
        write_training_history_csv(history_path, history)
        write_training_curves_svg(plot_path, history, best_epoch=best_epoch, run_title=f"V2 TCN-LTR training - {target.stem}")
        if stale >= patience:
            break
    return TemporalConvLTRScorer.load_checkpoint(target, device="cpu")[0]


def _parse_dilations(value: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value.split(",") if item)
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("dilations must be comma-separated positive integers")
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cache-dir", default="data/features_cache")
    parser.add_argument("--output", required=True)
    parser.add_argument("--training-log")
    parser.add_argument("--history-csv")
    parser.add_argument("--training-plot")
    parser.add_argument("--last-output", help="Checkpoint overwritten after every epoch for recovery.")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--dilations", type=_parse_dilations, default=(1, 2, 4, 8))
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--lambda-smooth", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-pairs-per-video", type=int, default=2048)
    args = parser.parse_args(argv)
    train(
        cache_dir=args.cache_dir,
        records=load_training_manifest(args.manifest, split=args.train_split),
        val_records=load_training_manifest(args.manifest, split=args.val_split),
        output=args.output,
        init_checkpoint=args.init_checkpoint,
        hidden_dim=args.hidden_dim,
        dilations=args.dilations,
        dropout=args.dropout,
        gamma=args.gamma,
        lambda_smooth=args.lambda_smooth,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        seed=args.seed,
        max_pairs_per_video=args.max_pairs_per_video,
        training_log=args.training_log,
        history_csv=args.history_csv,
        training_plot=args.training_plot,
        last_checkpoint=args.last_output,
    )


if __name__ == "__main__":
    main()
