"""Render transparent training-dynamics figures from the five CV fold histories."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

FOLD_COLORS = ["#0B3D91", "#D97706", "#0F766E", "#B45309", "#7C3AED"]


def read_history(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            {
                "epoch": float(row["epoch"]),
                "val_ap": float(row["val_ap"]),
                "train_total_loss": float(row["train_total_loss"]),
            }
            for row in csv.DictReader(handle)
        ]


def make_figure(report_dir: Path, output_dir: Path) -> None:
    histories = [
        read_history(report_dir / f"custom_fold{fold}_history.csv")
        for fold in range(5)
    ]

    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titleweight": "bold",
            "axes.labelcolor": "#14213D",
            "xtick.color": "#334155",
            "ytick.color": "#334155",
        }
    ):
        fig, (ax_ap, ax_loss) = plt.subplots(
            1,
            2,
            figsize=(13.2, 4.35),
            layout="constrained",
        )

        for fold, (history, color) in enumerate(zip(histories, FOLD_COLORS)):
            epochs = [row["epoch"] for row in history]
            validation_ap = [row["val_ap"] for row in history]
            train_loss = [row["train_total_loss"] for row in history]
            best_index = max(range(len(history)), key=lambda index: validation_ap[index])
            best_epoch = int(epochs[best_index])

            label = f"Fold {fold}  (best e={best_epoch})"
            ax_ap.plot(epochs, validation_ap, color=color, linewidth=2.0, label=label)
            ax_ap.scatter(
                [best_epoch],
                [validation_ap[best_index]],
                color=color,
                edgecolor="white",
                linewidth=1.2,
                marker="o",
                s=58,
                zorder=3,
            )
            ax_loss.plot(epochs, train_loss, color=color, linewidth=2.0, label=label)

        for axis in (ax_ap, ax_loss):
            axis.set_xlim(1, 50)
            axis.set_xticks([1, 10, 20, 30, 40, 50])
            axis.spines[["top", "right"]].set_visible(False)
            axis.grid(axis="y", color="#CBD5E1", linewidth=0.8, alpha=0.75)
            axis.set_xlabel("Epoch")

        ax_ap.set_title("Validation AP theo epoch", loc="left", color="#0B3D91")
        ax_ap.set_ylabel("Macro validation AP")
        ax_ap.set_ylim(0.30, 1.00)
        ax_ap.legend(
            loc="lower right",
            frameon=True,
            framealpha=0.96,
            edgecolor="#CBD5E1",
            fontsize=9,
        )
        ax_ap.text(
            0.0,
            -0.28,
            "Chấm tròn = checkpoint được chọn theo validation AP.",
            transform=ax_ap.transAxes,
            color="#475569",
            fontsize=9,
        )

        ax_loss.set_title("Train total loss theo epoch", loc="left", color="#0B3D91")
        ax_loss.set_ylabel("Pairwise margin + smoothness loss")
        ax_loss.text(
            0.0,
            -0.28,
            "Loss giảm xác nhận optimization hội tụ; không phải held-out metric.",
            transform=ax_loss.transAxes,
            color="#475569",
            fontsize=9,
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        for extension in ("svg", "png"):
            fig.savefig(
                output_dir / f"ltr_training_dynamics.{extension}",
                dpi=240,
                facecolor="white",
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=Path("data/reports"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("Cross-Domain/figures")
    )
    args = parser.parse_args()
    make_figure(args.report_dir, args.output_dir)


if __name__ == "__main__":
    main()
