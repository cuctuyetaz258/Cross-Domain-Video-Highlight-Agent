"""Train outer-specific nested OOF proposals and evaluate IMSAB on untouched test videos."""

from __future__ import annotations

import argparse
import json
import platform
import random
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from evaluation.evaluate_actionformer_ltr import evaluate_checkpoint
from highlight_agent.models.actionformer import ActionFormerConfig, load_actionformer_checkpoint
from highlight_agent.models.oof_proposals import build_nested_proposal_cache
from highlight_agent.models.proposal_ltr import ProposalLTRConfig
from highlight_agent.models.proposal_protocol import (
    assert_lineage_allowed,
    example_digest,
    split_contract,
)
from highlight_agent.models.train_actionformer_ltr import (
    ActionFormerExample,
    _atomic_json,
    load_actionformer_manifest,
    train_actionformer_localization,
    train_proposal_ltr,
)
from highlight_agent.models.training_artifacts import write_training_history_csv


def inner_splits(
    examples: list[ActionFormerExample], *, folds: int = 3, seed: int = 42
) -> list[tuple[list[ActionFormerExample], list[ActionFormerExample], list[ActionFormerExample]]]:
    """Fit/selection/prediction partitions; a prediction target never selects its own model."""
    if folds < 2 or len(examples) < max(6, folds * 2):
        raise ValueError("nested CV requires at least two inner folds and two videos per group")
    if len({x.video_id for x in examples}) != len(examples):
        raise ValueError("duplicate inner input video")
    groups: list[list[ActionFormerExample]] = [[] for _ in range(folds)]
    rng = random.Random(seed)
    cursor = 0
    for domain in sorted({x.domain for x in examples}):
        rows = sorted((x for x in examples if x.domain == domain), key=lambda x: x.video_id)
        rng.shuffle(rows)
        for row in rows:
            groups[cursor % folds].append(row)
            cursor += 1
    result = []
    for held_out in groups:
        targets = {x.video_id for x in held_out}
        remaining = sorted((x for x in examples if x.video_id not in targets), key=lambda x: x.video_id)
        rng.shuffle(remaining)
        selection_count = max(1, len(remaining) // 4)
        selection, fit = remaining[:selection_count], remaining[selection_count:]
        split_contract([x.video_id for x in fit], [x.video_id for x in selection], sorted(targets))
        result.append((fit, selection, held_out))
    return result


def artifact_paths(directory: Path) -> dict[str, Path]:
    return {
        "output_path": directory / "best.pt",
        "last_output_path": directory / "last.pt",
        "log_path": directory / "train_log.json",
        "history_csv_path": directory / "history.csv",
        "curves_path": directory / "curves.svg",
    }


def run_outer_fold(
    *,
    train: list[ActionFormerExample],
    val: list[ActionFormerExample],
    test: list[ActionFormerExample],
    directory: Path,
    fold: int,
    config: ActionFormerConfig,
    scorer_config: ProposalLTRConfig,
    inner_fold_count: int = 3,
    loc_epochs: int = 30,
    ltr_epochs: int = 30,
    patience: int = 8,
    seed: int = 42,
    device: str = "cpu",
    init_checkpoint: str | None = None,
    freeze_backbone: bool = False,
) -> dict[str, Any]:
    splits = split_contract([x.video_id for x in train], [x.video_id for x in val], [x.video_id for x in test])
    directory.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "status": "running",
        "outer_fold": fold,
        "outer_split": splits,
        "started_at_unix": time.time(),
        "stage": "inner_localization",
        "config": {
            "localization": config.to_dict(),
            "scorer": scorer_config.to_dict(),
            "seed": seed,
            "inner_folds": inner_fold_count,
            "loc_epochs": loc_epochs,
            "ltr_epochs": ltr_epochs,
            "patience": patience,
            "device": device,
        },
        "protocol": "nested_oof_v2_predicted_only_validation",
        "pretrained": False,
        "representation_policy": "outer_train_encoder_frozen_for_ltr; proposal-only cross-fitting",
        "content_fingerprints": {x.video_id: example_digest(x) for x in train + val + test},
        "inner_runs": [],
    }
    if init_checkpoint:
        report["pretrained"] = True
        report["initialization"] = {
            "checkpoint": str(Path(init_checkpoint).resolve()),
            "freeze_backbone": freeze_backbone,
        }
    log_path = directory / "run_report.json"
    _atomic_json(log_path, report)
    active_log: Path | None = None
    try:
        generators = []
        ancestors = []
        budgets = []
        for index, (fit, selection, targets) in enumerate(inner_splits(train, folds=inner_fold_count, seed=seed)):
            paths = artifact_paths(directory / f"inner{index}")
            active_log = paths["log_path"]
            print(f"fold={fold} inner={index} fit={len(fit)} selection={len(selection)} OOF={len(targets)}", flush=True)
            _, inner_report = train_actionformer_localization(
                train_examples=fit,
                val_examples=selection,
                config=config,
                **paths,
                max_epochs=loc_epochs,
                patience=patience,
                seed=seed,
                device=device,
                run_name=f"nested_fold{fold}_inner{index}",
                init_checkpoint_path=init_checkpoint,
                freeze_backbone=freeze_backbone,
            )
            _, metadata, _ = load_actionformer_checkpoint(paths["output_path"])
            assert_lineage_allowed(metadata["data_lineage"], set(splits["train"]) - {x.video_id for x in targets})
            generators.append((paths["output_path"], targets))
            ancestors.append(metadata["data_lineage"])
            budgets.append(inner_report["best_epoch"])
            report["inner_runs"].append(
                {
                    "inner_fold": index,
                    "best_epoch": inner_report["best_epoch"],
                    "train": sorted(x.video_id for x in fit),
                    "selection": sorted(x.video_id for x in selection),
                    "prediction": sorted(x.video_id for x in targets),
                    "checkpoint": str(paths["output_path"]),
                }
            )
            _atomic_json(log_path, report)
        fixed_budget = max(1, int(statistics.median(budgets)))
        report.update(stage="outer_refit", outer_epoch_budget=fixed_budget)
        _atomic_json(log_path, report)
        paths = artifact_paths(directory / "outer_localization")
        active_log = paths["log_path"]
        print(f"fold={fold} refit epochs={fixed_budget} (selected inside outer train)", flush=True)
        model, _ = train_actionformer_localization(
            train_examples=train,
            val_examples=[],
            config=config,
            **paths,
            max_epochs=fixed_budget,
            patience=patience,
            seed=seed,
            device=device,
            run_name=f"nested_fold{fold}_outer_refit",
            fixed_epochs=True,
            ancestor_lineage=ancestors,
            init_checkpoint_path=init_checkpoint,
            freeze_backbone=freeze_backbone,
        )
        outer_checkpoint = paths["output_path"]
        _, metadata, _ = load_actionformer_checkpoint(outer_checkpoint)
        report.update(stage="nested_cache")
        _atomic_json(log_path, report)
        cache = build_nested_proposal_cache(
            outer_fold=fold,
            train_examples=train,
            val_video_ids=splits["val"],
            test_video_ids=splits["test"],
            generators=generators,
            outer_checkpoint=outer_checkpoint,
            output_path=directory / "nested_proposals.json",
            device=device,
        )
        report["proposal_counts"] = {key: len(row["proposals"]) for key, row in cache["videos"].items()}
        report.update(stage="imsab_training")
        _atomic_json(log_path, report)
        paths = artifact_paths(directory / "imsab")
        active_log = paths["log_path"]
        print(f"fold={fold} IMSAB training; validation candidates=predicted_only", flush=True)
        _, ltr_report = train_proposal_ltr(
            actionformer=model,
            checkpoint_metadata=metadata,
            train_examples=train,
            val_examples=val,
            **paths,
            max_epochs=ltr_epochs,
            learning_rate=2e-4,
            patience=patience,
            scorer_config=scorer_config,
            seed=seed,
            device=device,
            run_name=f"nested_fold{fold}_imsab",
            nested_cache_path=directory / "nested_proposals.json",
            source_checkpoint_path=outer_checkpoint,
            outer_test_video_ids=splits["test"],
        )
        report.update(stage="held_out_evaluation", best_val_ndcg_at_3=ltr_report["best_val_ndcg_at_3"])
        _atomic_json(log_path, report)
        allowed = set(splits["train"] + splits["val"])
        assert_lineage_allowed(ltr_report["checkpoint_metadata"]["data_lineage"], allowed)
        for name, checkpoint in (("confidence", outer_checkpoint), ("imsab", paths["output_path"])):
            evaluation = evaluate_checkpoint(checkpoint, test, device=device)
            evaluation.update(split="test", outer_fold=fold, protocol=report["protocol"])
            _atomic_json(directory / f"{name}_evaluation_test.json", evaluation)
            report[f"{name}_test_metrics"] = evaluation["metrics"]
        report.update(status="complete", stage="complete", completed_at_unix=time.time())
        report["elapsed_seconds"] = report["completed_at_unix"] - report["started_at_unix"]
        _atomic_json(log_path, report)
        return report
    except BaseException as exc:
        report.update(
            status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            error=f"{type(exc).__name__}: {exc}",
            updated_at_unix=time.time(),
        )
        _atomic_json(log_path, report)
        if active_log is not None and active_log.is_file():
            active = json.loads(active_log.read_text(encoding="utf-8"))
            if active.get("status") == "running":
                active.update(status=report["status"], error=report["error"])
                _atomic_json(active_log, active)
        raise


def summarize(directory: Path, reports: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for report in reports:
        row = {"fold": report["outer_fold"], "val_predicted_ndcg3": report["best_val_ndcg_at_3"]}
        for family in ("confidence", "imsab"):
            row.update({f"{family}_{key}": value for key, value in report[f"{family}_test_metrics"].items()})
        rows.append(row)
    summary = {
        key: {"mean": statistics.mean(row[key] for row in rows), "std": statistics.pstdev(row[key] for row in rows)}
        for key in rows[0]
        if key != "fold"
    }
    _atomic_json(
        directory / "cv_summary.json", {"folds": rows, "summary": summary, "aggregation": "macro fold population SD"}
    )
    write_training_history_csv(directory / "cv_metrics.csv", rows)
    lines = [
        "# Nested OOF v2 training report",
        "",
        "Protocol: separate inner fit/selection/prediction partitions per outer fold; validation predicted-only.",
        "",
        "These are scratch ActionFormer runs, not TVSum/SumMe pretraining. The outer encoder is trained only on outer train and frozen for LTR; OOF applies to proposal coordinates/confidence, not the pooled encoder features.",
        "",
        "Old shared-OOF/GT-candidate metrics are not a comparable clean baseline. The confidence baseline below uses the same outer generator and test candidates as IMSAB.",
        "",
        "| Fold | Val predicted nDCG@3 | Confidence test mAP@0.3 | IMSAB test mAP@0.3 | IMSAB Recall@3 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['fold']} | {row['val_predicted_ndcg3']:.4f} | {row['confidence_map_iou_0_3']:.4f} | {row['imsab_map_iou_0_3']:.4f} | {row['imsab_recall_at_3_iou_0_3']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Every inner localization, outer refit and IMSAB run includes train_log.json, history.csv, curves.svg, best.pt and last.pt. The fixed-budget outer refit has no validation score by design.",
            "",
            "Single-seed results on an already explored 18-video dataset are not an independent confirmation set. Multi-seed replication and a new holdout are required before promotion.",
        ]
    )
    (directory / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-template", default="data/manifests/actionformer_fold{fold}.jsonl")
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--output-dir", required=True, help="Must be a new directory; previous runs are never overwritten."
    )
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--loc-epochs", type=int, default=30)
    parser.add_argument("--ltr-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--init-checkpoint", default=None, help="Benchmark-pretrained ActionFormer checkpoint.")
    parser.add_argument(
        "--freeze-backbone", action="store_true", help="Keep the shared pretrained backbone frozen during localization."
    )
    args = parser.parse_args(argv)
    directory = Path(args.output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    if args.init_checkpoint:
        initialized_model, _, _ = load_actionformer_checkpoint(args.init_checkpoint, device="cpu")
        localization_config = initialized_model.config
    else:
        localization_config = ActionFormerConfig(d_model=32, attention_window=64)
    _atomic_json(
        directory / "environment.json",
        {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "arguments": vars(args),
        },
    )
    reports = []
    for fold in args.folds:
        manifest = args.manifest_template.format(fold=fold)
        data = {
            split: load_actionformer_manifest(manifest, split=split, project_root=args.project_root, ready_only=False)
            for split in ("train", "val", "test")
        }
        reports.append(
            run_outer_fold(
                **data,
                directory=directory / f"fold{fold}",
                fold=fold,
                config=localization_config,
                scorer_config=ProposalLTRConfig(),
                inner_fold_count=args.inner_folds,
                loc_epochs=args.loc_epochs,
                ltr_epochs=args.ltr_epochs,
                patience=args.patience,
                seed=args.seed,
                device=args.device,
                init_checkpoint=args.init_checkpoint,
                freeze_backbone=args.freeze_backbone,
            )
        )
        summarize(directory, reports)
    print(json.dumps({"status": "complete", "folds": args.folds, "report": str(directory / "REPORT.md")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
