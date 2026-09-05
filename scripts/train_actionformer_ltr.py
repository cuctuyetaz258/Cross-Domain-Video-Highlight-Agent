"""Train the ActionFormer localization stage and persist report-ready logs."""

from __future__ import annotations

import argparse
import json

from highlight_agent.models.actionformer import ActionFormerConfig, load_actionformer_checkpoint
from highlight_agent.models.oof_proposals import load_oof_proposal_cache
from highlight_agent.models.proposal_ltr import ProposalLTRConfig
from highlight_agent.models.train_actionformer_ltr import (
    load_actionformer_manifest,
    train_actionformer_localization,
    train_proposal_ltr,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/manifests/actionformer_fold0.jsonl")
    parser.add_argument("--stage", choices=["localization", "ltr"], default="localization")
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output", default="data/models/actionformer_localization.pt")
    parser.add_argument("--last-output", default="data/models/actionformer_localization_last.pt")
    parser.add_argument("--log", default="data/reports/actionformer_training_log.json")
    parser.add_argument("--history-csv", default="data/reports/actionformer_training_history.csv")
    parser.add_argument("--curves", default="data/reports/actionformer_training_curves.svg")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-reg", type=float, default=1.0)
    parser.add_argument("--lambda-smooth", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--attention-window", type=int, default=128)
    parser.add_argument("--max-train-videos", type=int, default=None)
    parser.add_argument("--max-val-videos", type=int, default=None)
    parser.add_argument("--run-name", default="actionformer_localization")
    parser.add_argument("--ltr-architecture", choices=["setrank_imsab", "mlp"], default="setrank_imsab")
    parser.add_argument("--pairwise-loss", choices=["margin", "ranknet"], default="ranknet")
    parser.add_argument("--pair-weighting", choices=["none", "utility", "delta_ndcg"], default="utility")
    parser.add_argument("--num-imsab-blocks", type=int, default=2)
    parser.add_argument("--num-inducing-points", type=int, default=16)
    parser.add_argument("--ltr-num-heads", type=int, default=2)
    parser.add_argument("--ltr-ffn-dim", type=int, default=256)
    parser.add_argument("--ltr-dropout", type=float, default=0.3)
    parser.add_argument("--rank-signal", choices=["none", "actionformer_ordinal"], default="actionformer_ordinal")
    parser.add_argument("--utility-delta", type=float, default=0.1)
    parser.add_argument("--ndcg-k", type=int, default=3)
    parser.add_argument("--gain-scale", type=float, default=4.0)
    parser.add_argument("--max-pairs-per-video", type=int, default=256)
    parser.add_argument("--proposal-cache", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    train_examples = load_actionformer_manifest(
        args.manifest,
        split="train",
        project_root=args.project_root,
    )
    val_examples = load_actionformer_manifest(
        args.manifest,
        split="val",
        project_root=args.project_root,
    )
    if args.max_train_videos is not None:
        train_examples = train_examples[: args.max_train_videos]
    if args.max_val_videos is not None:
        val_examples = val_examples[: args.max_val_videos]
    if args.stage == "localization":
        config = ActionFormerConfig(
            d_model=args.d_model,
            attention_window=args.attention_window,
        )
        _, report = train_actionformer_localization(
            train_examples=train_examples,
            val_examples=val_examples,
            output_path=args.output,
            last_output_path=args.last_output,
            log_path=args.log,
            history_csv_path=args.history_csv,
            curves_path=args.curves,
            config=config,
            max_epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            lambda_reg=args.lambda_reg,
            lambda_smooth=args.lambda_smooth,
            patience=args.patience,
            seed=args.seed,
            device=args.device,
            run_name=args.run_name,
        )
    else:
        if not args.init_checkpoint:
            raise ValueError("--init-checkpoint is required for --stage ltr")
        actionformer, metadata, _ = load_actionformer_checkpoint(
            args.init_checkpoint,
            device=args.device or "cpu",
        )
        proposal_cache = None
        proposal_cache_metadata = None
        if args.proposal_cache:
            proposal_cache, proposal_cache_metadata = load_oof_proposal_cache(args.proposal_cache)
        _, report = train_proposal_ltr(
            actionformer=actionformer,
            checkpoint_metadata=metadata,
            train_examples=train_examples,
            val_examples=val_examples,
            output_path=args.output,
            last_output_path=args.last_output,
            log_path=args.log,
            history_csv_path=args.history_csv,
            curves_path=args.curves,
            max_epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            utility_delta=args.utility_delta,
            scorer_config=ProposalLTRConfig(
                architecture=args.ltr_architecture,
                d_model=args.d_model,
                num_imsab_blocks=args.num_imsab_blocks,
                num_inducing_points=args.num_inducing_points,
                num_heads=args.ltr_num_heads,
                ffn_dim=args.ltr_ffn_dim,
                dropout=args.ltr_dropout,
                rank_signal=args.rank_signal,
            ),
            loss_type=args.pairwise_loss,
            pair_weighting=args.pair_weighting,
            ndcg_k=args.ndcg_k,
            gain_scale=args.gain_scale,
            max_pairs_per_video=args.max_pairs_per_video,
            patience=args.patience,
            seed=args.seed,
            device=args.device,
            run_name=args.run_name,
            predicted_proposals_by_video=proposal_cache,
            proposal_cache_metadata=proposal_cache_metadata,
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
