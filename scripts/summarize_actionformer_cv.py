"""Aggregate ActionFormer/IMSAB cross-validation logs into report artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

METRICS = (
    "localization_val_recall_at_3_iou_0_3",
    "ltr_val_ndcg_at_3",
    "map_iou_0_3",
    "map_iou_0_5",
    "map_iou_0_7",
    "recall_at_3_iou_0_3",
    "recall_at_5_iou_0_3",
    "mean_iou",
    "seconds_per_video",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def collect_folds(report_dir: Path, folds: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fold in range(folds):
        localization = _read_json(report_dir / f"fold{fold}_localization_log.json")
        ltr = _read_json(report_dir / f"fold{fold}_imsab_oof_log.json")
        evaluation = _read_json(report_dir / f"fold{fold}_imsab_oof_evaluation_test.json")
        if localization.get("status") != "complete" or ltr.get("status") != "complete":
            raise ValueError(f"fold {fold} training report is incomplete")
        metrics = evaluation["metrics"]
        rows.append(
            {
                "fold": fold,
                "localization_best_epoch": localization["best_epoch"],
                "localization_epochs": len(localization["epochs"]),
                "localization_val_recall_at_3_iou_0_3": localization[
                    "best_val_recall_at_3_iou_0_3"
                ],
                "ltr_best_epoch": ltr["best_epoch"],
                "ltr_epochs": len(ltr["epochs"]),
                "ltr_val_ndcg_at_3": ltr["best_val_ndcg_at_3"],
                "map_iou_0_3": metrics["map_iou_0_3"],
                "map_iou_0_5": metrics["map_iou_0_5"],
                "map_iou_0_7": metrics["map_iou_0_7"],
                "recall_at_3_iou_0_3": metrics["recall_at_3_iou_0_3"],
                "recall_at_5_iou_0_3": metrics["recall_at_5_iou_0_3"],
                "mean_iou": metrics["mean_iou"],
                "duration_valid_rate": metrics["duration_valid_rate"],
                "seconds_per_video": evaluation["seconds_per_video"],
                "test_video_count": evaluation["video_count"],
            }
        )
    return rows


def aggregate(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    return {
        metric: {
            "mean": mean(float(row[metric]) for row in rows),
            "std": pstdev(float(row[metric]) for row in rows),
        }
        for metric in METRICS
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_svg(path: Path, rows: list[dict[str, Any]]) -> None:
    width, height = 920, 460
    left, top, plot_height = 70, 45, 330
    series = (
        ("ltr_val_ndcg_at_3", "Validation nDCG@3", "#2563eb"),
        ("map_iou_0_3", "Test mAP@0.3", "#f97316"),
        ("recall_at_3_iou_0_3", "Test Recall@3", "#16a34a"),
    )
    group_width = 150
    bar_width = 30
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="70" y="26" font-family="Arial" font-size="18" font-weight="bold">ActionFormer + IMSAB OOF · 5-fold results</text>',
    ]
    for tick in range(6):
        value = tick / 5
        y = top + plot_height * (1 - value)
        elements.append(f'<line x1="{left}" y1="{y:.1f}" x2="850" y2="{y:.1f}" stroke="#e5e7eb"/>')
        elements.append(f'<text x="38" y="{y + 4:.1f}" font-family="Arial" font-size="11">{value:.1f}</text>')
    for index, row in enumerate(rows):
        group_x = left + 55 + index * group_width
        for series_index, (metric, _, color) in enumerate(series):
            value = max(0.0, min(float(row[metric]), 1.0))
            bar_height = value * plot_height
            x = group_x + series_index * (bar_width + 5)
            y = top + plot_height - bar_height
            elements.append(
                f'<rect x="{x}" y="{y:.1f}" width="{bar_width}" height="{bar_height:.1f}" fill="{color}" rx="2"/>'
            )
        elements.append(
            f'<text x="{group_x + 48}" y="400" text-anchor="middle" font-family="Arial" font-size="12">Fold {row["fold"]}</text>'
        )
    legend_x = 565
    for index, (_, label, color) in enumerate(series):
        y = 420 + index * 16
        elements.append(f'<rect x="{legend_x}" y="{y - 10}" width="11" height="11" fill="{color}"/>')
        elements.append(f'<text x="{legend_x + 17}" y="{y}" font-family="Arial" font-size="11">{label}</text>')
    elements.append("</svg>")
    _atomic_text(path, "\n".join(elements))


def write_markdown(
    path: Path,
    rows: list[dict[str, Any]],
    summary: dict[str, dict[str, float]],
) -> None:
    lines = [
        "# ActionFormer + IMSAB OOF: 5-fold Training Report",
        "",
        "## Configuration",
        "",
        "- ActionFormer localization: `d_model=32`, attention window 64, seed 42.",
        "- Proposal scorer: two IMSAB blocks, 16 inducing points, `d_model=128`, two heads.",
        "- Objective: utility-weighted RankNet using `proposal_utility_v2`.",
        "- Candidate source: leakage-checked OOF proposal cache covering all 18 videos.",
        "- Execution device: CPU after CUDA OOM on the full proposal lists.",
        "",
        "## Per-fold results",
        "",
        "| Fold | Loc. val Recall@3 | LTR val nDCG@3 | Test mAP@0.3 | Test Recall@3 | Test mean IoU |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f'| {row["fold"]} | {row["localization_val_recall_at_3_iou_0_3"]:.4f} | '
            f'{row["ltr_val_ndcg_at_3"]:.4f} | {row["map_iou_0_3"]:.4f} | '
            f'{row["recall_at_3_iou_0_3"]:.4f} | {row["mean_iou"]:.4f} |'
        )
    lines.extend(
        [
            "",
            "## Cross-validation summary",
            "",
            "Values are the unweighted mean and population standard deviation across five folds.",
            "",
            f'- Localization validation Recall@3: `{summary["localization_val_recall_at_3_iou_0_3"]["mean"]:.4f} ± {summary["localization_val_recall_at_3_iou_0_3"]["std"]:.4f}`.',
            f'- LTR validation nDCG@3: `{summary["ltr_val_ndcg_at_3"]["mean"]:.4f} ± {summary["ltr_val_ndcg_at_3"]["std"]:.4f}`.',
            f'- Test mAP@0.3: `{summary["map_iou_0_3"]["mean"]:.4f} ± {summary["map_iou_0_3"]["std"]:.4f}`.',
            f'- Test mAP@0.5: `{summary["map_iou_0_5"]["mean"]:.4f} ± {summary["map_iou_0_5"]["std"]:.4f}`.',
            f'- Test Recall@3: `{summary["recall_at_3_iou_0_3"]["mean"]:.4f} ± {summary["recall_at_3_iou_0_3"]["std"]:.4f}`.',
            f'- Test mean IoU: `{summary["mean_iou"]["mean"]:.4f} ± {summary["mean_iou"]["std"]:.4f}`.',
            "",
            "## Interpretation",
            "",
            "The scorer learns the validation ranking signal, but localization remains the bottleneck: test mAP and IoU are low and vary substantially by fold. These results do not pass the production gate. The next experiment should compare the same OOF candidates against ActionFormer confidence and the legacy MLP baseline before changing the IMSAB architecture.",
        ]
    )
    _atomic_text(path, "\n".join(lines) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", default="data/reports/actionformer_cv")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output-json", default="data/reports/actionformer_cv/cv_summary.json")
    parser.add_argument("--output-csv", default="data/reports/actionformer_cv/cv_fold_metrics.csv")
    parser.add_argument("--output-svg", default="data/reports/actionformer_cv/cv_metrics.svg")
    parser.add_argument("--output-markdown", default="ACTIONFORMER_CV_REPORT.md")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = collect_folds(Path(args.report_dir), args.folds)
    summary = aggregate(rows)
    _atomic_text(
        Path(args.output_json),
        json.dumps({"fold_count": len(rows), "folds": rows, "summary": summary}, indent=2) + "\n",
    )
    write_csv(Path(args.output_csv), rows)
    write_svg(Path(args.output_svg), rows)
    write_markdown(Path(args.output_markdown), rows, summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
