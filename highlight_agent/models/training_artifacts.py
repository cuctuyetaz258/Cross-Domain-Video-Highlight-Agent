"""Dependency-free training-history exports for reports and reproducibility."""

from __future__ import annotations

import csv
import math
from html import escape
from pathlib import Path
from typing import Any, Iterable


def _atomic_text_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(content, encoding="utf-8", newline="")
    temporary_path.replace(path)


def _flatten_epoch(epoch: dict[str, Any]) -> dict[str, Any]:
    row = {
        key: value
        for key, value in epoch.items()
        if not isinstance(value, (dict, list, tuple))
    }
    for group_key, prefix in (
        ("ap_by_domain", "ap_domain"),
        ("ap_by_source", "ap_source"),
        ("ap_by_video", "ap_video"),
        ("source_pair_counts", "pairs_source"),
    ):
        for name, value in sorted((epoch.get(group_key) or {}).items()):
            row[f"{prefix}__{name}"] = value
    return row


def write_training_history_csv(path: str | Path, epochs: Iterable[dict[str, Any]]) -> None:
    """Write a flat, spreadsheet-friendly history without losing group metrics."""

    rows = [_flatten_epoch(epoch) for epoch in epochs]
    if not rows:
        return
    preferred = [
        "epoch",
        "train_margin_loss",
        "train_smooth_loss",
        "train_total_loss",
        "selection_ap",
        "val_ap",
        "train_ap",
        "learning_rate",
        "selection_split",
        "selection_metric",
    ]
    all_fields = {field for row in rows for field in row}
    fieldnames = [field for field in preferred if field in all_fields]
    fieldnames.extend(sorted(all_fields - set(fieldnames)))

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(destination.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(destination)


def _numeric_series(
    epochs: list[dict[str, Any]], key: str
) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for item in epochs:
        value = item.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            points.append((float(item["epoch"]), float(value)))
    return points


def _group_series(
    epochs: list[dict[str, Any]], group_key: str, prefix: str
) -> dict[str, list[tuple[float, float]]]:
    names = sorted({name for item in epochs for name in (item.get(group_key) or {})})
    return {
        f"{prefix}: {name}": [
            (float(item["epoch"]), float(item[group_key][name]))
            for item in epochs
            if name in (item.get(group_key) or {})
        ]
        for name in names
    }


def _panel_svg(
    *,
    title: str,
    series: dict[str, list[tuple[float, float]]],
    x: float,
    y: float,
    width: float,
    height: float,
    colors: list[str],
    best_epoch: int | None = None,
) -> str:
    plot_left = x + 65
    plot_top = y + 34
    plot_width = width - 90
    plot_height = height - 78
    points = [point for values in series.values() for point in values]
    if not points:
        return ""
    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    if x_min == x_max:
        x_min -= 0.5
        x_max += 0.5
    if y_min == y_max:
        padding = max(abs(y_min) * 0.1, 0.05)
        y_min -= padding
        y_max += padding
    else:
        padding = (y_max - y_min) * 0.08
        y_min -= padding
        y_max += padding

    def sx(value: float) -> float:
        return plot_left + (value - x_min) / (x_max - x_min) * plot_width

    def sy(value: float) -> float:
        return plot_top + plot_height - (value - y_min) / (y_max - y_min) * plot_height

    output = [
        f'<text x="{x + 8}" y="{y + 20}" class="panel-title">{escape(title)}</text>',
        f'<rect x="{plot_left}" y="{plot_top}" width="{plot_width}" height="{plot_height}" class="plot"/>',
    ]
    for tick in range(5):
        ratio = tick / 4
        tick_y = plot_top + ratio * plot_height
        tick_value = y_max - ratio * (y_max - y_min)
        output.extend(
            [
                f'<line x1="{plot_left}" y1="{tick_y:.2f}" x2="{plot_left + plot_width}" y2="{tick_y:.2f}" class="grid"/>',
                f'<text x="{plot_left - 8}" y="{tick_y + 4:.2f}" class="tick" text-anchor="end">{tick_value:.4g}</text>',
            ]
        )
    output.append(
        f'<text x="{plot_left + plot_width / 2}" y="{plot_top + plot_height + 35}" class="axis" text-anchor="middle">Epoch</text>'
    )
    output.append(
        f'<text x="{plot_left}" y="{plot_top + plot_height + 18}" class="tick">{x_min:g}</text>'
    )
    output.append(
        f'<text x="{plot_left + plot_width}" y="{plot_top + plot_height + 18}" class="tick" text-anchor="end">{x_max:g}</text>'
    )
    if best_epoch is not None and x_min <= best_epoch <= x_max:
        best_x = sx(float(best_epoch))
        output.append(
            f'<line x1="{best_x:.2f}" y1="{plot_top}" x2="{best_x:.2f}" y2="{plot_top + plot_height}" class="best"/>'
        )
        output.append(
            f'<text x="{best_x + 4:.2f}" y="{plot_top + 12}" class="best-label">best={best_epoch}</text>'
        )
    legend_x = plot_left + 8
    legend_y = plot_top + 16
    for index, (name, values) in enumerate(series.items()):
        if not values:
            continue
        color = colors[index % len(colors)]
        coordinates = " ".join(f"{sx(px):.2f},{sy(py):.2f}" for px, py in values)
        output.append(
            f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="2.2"/>'
        )
        output.append(
            f'<circle cx="{sx(values[-1][0]):.2f}" cy="{sy(values[-1][1]):.2f}" r="2.8" fill="{color}"/>'
        )
        legend_row = index % 4
        legend_column = index // 4
        lx = legend_x + legend_column * 210
        ly = legend_y + legend_row * 17
        output.append(f'<line x1="{lx}" y1="{ly - 4}" x2="{lx + 18}" y2="{ly - 4}" stroke="{color}" stroke-width="2.2"/>')
        output.append(f'<text x="{lx + 23}" y="{ly}" class="legend">{escape(name)}</text>')
    return "\n".join(output)


def write_training_curves_svg(
    path: str | Path,
    epochs: Iterable[dict[str, Any]],
    *,
    best_epoch: int | None,
    run_title: str = "LTR training history",
) -> None:
    """Write a report-ready SVG containing loss, AP and learning-rate curves."""

    history = list(epochs)
    if not history:
        return
    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2"]
    loss_series = {
        "Total loss": _numeric_series(history, "train_total_loss"),
        "Margin loss": _numeric_series(history, "train_margin_loss"),
        "Smooth loss": _numeric_series(history, "train_smooth_loss"),
    }
    ap_series = {"Selection AP": _numeric_series(history, "selection_ap")}
    ap_series.update(_group_series(history, "ap_by_domain", "Domain"))
    ap_series.update(_group_series(history, "ap_by_source", "Source"))
    lr_series = {"Learning rate": _numeric_series(history, "learning_rate")}
    panels = [
        _panel_svg(
            title="Training losses",
            series=loss_series,
            x=20,
            y=72,
            width=1160,
            height=300,
            colors=colors,
            best_epoch=best_epoch,
        ),
        _panel_svg(
            title="Validation/selection Average Precision",
            series=ap_series,
            x=20,
            y=390,
            width=1160,
            height=300,
            colors=colors,
            best_epoch=best_epoch,
        ),
        _panel_svg(
            title="Learning-rate schedule",
            series=lr_series,
            x=20,
            y=708,
            width=1160,
            height=260,
            colors=colors,
            best_epoch=best_epoch,
        ),
    ]
    subtitle = f"Epochs recorded: {len(history)} · Best epoch: {best_epoch or 'N/A'}"
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="990" viewBox="0 0 1200 990">
<style>
  .background {{ fill: #ffffff; }}
  .plot {{ fill: #f8fafc; stroke: #cbd5e1; }}
  .grid {{ stroke: #e2e8f0; stroke-width: 1; }}
  .title {{ font: 700 24px system-ui, sans-serif; fill: #0f172a; }}
  .subtitle {{ font: 14px system-ui, sans-serif; fill: #475569; }}
  .panel-title {{ font: 700 16px system-ui, sans-serif; fill: #1e293b; }}
  .axis {{ font: 12px system-ui, sans-serif; fill: #334155; }}
  .tick {{ font: 11px system-ui, sans-serif; fill: #64748b; }}
  .legend {{ font: 11px system-ui, sans-serif; fill: #334155; }}
  .best {{ stroke: #0f172a; stroke-width: 1.5; stroke-dasharray: 5 4; }}
  .best-label {{ font: 700 10px system-ui, sans-serif; fill: #0f172a; }}
</style>
<rect width="1200" height="990" class="background"/>
<text x="28" y="34" class="title">{escape(run_title)}</text>
<text x="28" y="56" class="subtitle">{escape(subtitle)}</text>
{''.join(panels)}
</svg>
"""
    _atomic_text_write(Path(path), svg)
