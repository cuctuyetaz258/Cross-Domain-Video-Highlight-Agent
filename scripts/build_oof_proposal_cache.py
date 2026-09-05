"""Build a leakage-safe ActionFormer out-of-fold proposal cache."""

from __future__ import annotations

import argparse
import json

from highlight_agent.models.oof_proposals import build_oof_proposal_cache


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--manifest-template", default="data/manifests/actionformer_fold{fold}.jsonl")
    parser.add_argument(
        "--checkpoint-template",
        default="data/models/actionformer_fold{fold}_localization_cv.pt",
    )
    parser.add_argument("--output", default="data/proposals/actionformer_oof_v1.json")
    parser.add_argument("--stats-csv", default="data/reports/actionformer_cv/oof_proposal_stats.csv")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    inputs = [
        (
            fold,
            args.manifest_template.format(fold=fold),
            args.checkpoint_template.format(fold=fold),
        )
        for fold in range(args.folds)
    ]
    report = build_oof_proposal_cache(
        fold_inputs=inputs,
        output_path=args.output,
        stats_csv_path=args.stats_csv,
        project_root=args.project_root,
        device=args.device,
    )
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
