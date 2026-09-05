"""Rerank fixed nested-CV proposals without retraining localization."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
from pathlib import Path
from typing import Any

import torch

# Support the documented `python scripts/...` invocation from a checkout.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.evaluate_actionformer_ltr import evaluate_checkpoint
from highlight_agent.models.actionformer import load_actionformer_checkpoint
from highlight_agent.models.proposal_ltr import ProposalLTRConfig
from highlight_agent.models.train_actionformer_ltr import (
    _atomic_json,
    load_actionformer_manifest,
    train_proposal_ltr,
)
from highlight_agent.models.training_artifacts import write_training_history_csv


def _paths(directory: Path) -> dict[str, Path]:
    return {
        "output_path": directory / "best.pt",
        "last_output_path": directory / "last.pt",
        "log_path": directory / "train_log.json",
        "history_csv_path": directory / "history.csv",
        "curves_path": directory / "curves.svg",
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    metric_names = sorted(rows[0]["test_metrics"])
    return {
        name: {
            "mean": statistics.mean(row["test_metrics"][name] for row in rows),
            "std": statistics.pstdev(row["test_metrics"][name] for row in rows),
        }
        for name in metric_names
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True, help="Completed nested-CV directory with foldN caches.")
    parser.add_argument("--output-dir", required=True, help="New directory for this ablation; never overwritten.")
    parser.add_argument("--manifest-template", default="data/manifests/actionformer_fold{fold}.jsonl")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--architecture", choices=["mlp", "setrank_imsab"], required=True)
    parser.add_argument("--loss", choices=["margin", "ranknet"], required=True)
    parser.add_argument("--pair-weighting", choices=["none", "utility", "delta_ndcg"], default="utility")
    parser.add_argument("--rank-signal", choices=["none", "actionformer_ordinal"], default="none")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-imsab-blocks", type=int, default=2)
    parser.add_argument("--num-inducing-points", type=int, default=16)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--ffn-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source = Path(args.source_run).resolve()
    destination = Path(args.output_dir).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"source run does not exist: {source}")
    destination.mkdir(parents=True, exist_ok=False)
    config = ProposalLTRConfig(
        architecture=args.architecture,
        d_model=args.d_model,
        num_imsab_blocks=args.num_imsab_blocks,
        num_inducing_points=args.num_inducing_points,
        num_heads=args.num_heads,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
        rank_signal=args.rank_signal,
    )
    _atomic_json(
        destination / "environment.json",
        {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "arguments": vars(args),
            "source_run": str(source),
            "scorer": config.to_dict(),
            "protocol": "fixed_generator_nested_oof_v2_predicted_only",
        },
    )
    rows: list[dict[str, Any]] = []
    for fold in args.folds:
        source_fold = source / f"fold{fold}"
        checkpoint = source_fold / "outer_localization" / "best.pt"
        proposal_cache = source_fold / "nested_proposals.json"
        if not checkpoint.is_file() or not proposal_cache.is_file():
            raise FileNotFoundError(f"fold {fold} is missing its fixed localization checkpoint or proposal cache")
        manifest = args.manifest_template.format(fold=fold)
        examples = {
            split: load_actionformer_manifest(manifest, split=split, project_root=args.project_root)
            for split in ("train", "val", "test")
        }
        model, metadata, _ = load_actionformer_checkpoint(checkpoint, device=args.device)
        fold_dir = destination / f"fold{fold}"
        print(f"fold={fold} reranking fixed nested proposals", flush=True)
        _, report = train_proposal_ltr(
            actionformer=model,
            checkpoint_metadata=metadata,
            train_examples=examples["train"],
            val_examples=examples["val"],
            **_paths(fold_dir),
            max_epochs=args.epochs,
            learning_rate=2e-4,
            patience=args.patience,
            scorer_config=config,
            loss_type=args.loss,
            pair_weighting=args.pair_weighting,
            seed=args.seed,
            device=args.device,
            run_name=f"rerank_ablation_fold{fold}",
            nested_cache_path=proposal_cache,
            source_checkpoint_path=checkpoint,
            outer_test_video_ids=[item.video_id for item in examples["test"]],
        )
        evaluation = evaluate_checkpoint(_paths(fold_dir)["output_path"], examples["test"], device=args.device)
        _atomic_json(fold_dir / "evaluation_test.json", evaluation)
        rows.append(
            {
                "fold": fold,
                "best_val_ndcg_at_3": report["best_val_ndcg_at_3"],
                "test_metrics": evaluation["metrics"],
            }
        )
        _atomic_json(destination / "cv_summary.json", {"folds": rows, "summary": _summary(rows)})
    write_training_history_csv(
        destination / "cv_metrics.csv",
        [
            {"fold": row["fold"], "best_val_ndcg_at_3": row["best_val_ndcg_at_3"], **row["test_metrics"]}
            for row in rows
        ],
    )
    print(json.dumps({"status": "complete", "output_dir": str(destination), "summary": _summary(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
