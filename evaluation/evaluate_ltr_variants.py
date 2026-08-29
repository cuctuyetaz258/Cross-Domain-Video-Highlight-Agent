"""Report full LTR, channel sensitivity, LTR+LLM, and LLM failure separately."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from evaluation.evaluate_ltr import _measure_scores, _method_result
from highlight_agent.models.ltr_scorer import AdditiveAttentionScorer
from highlight_agent.models.train_offline import (
    FEATURE_CHANNELS,
    build_window_examples,
    load_training_manifest,
)

FULL_LTR_VARIANT = "ltr_full_7ch"
LLM_SUCCESS_VARIANT = "ltr_llm_rerank"
LLM_FAILURE_VARIANT = "ltr_llm_failure"
CHANNEL_INDEX = {name: index for index, name in enumerate(FEATURE_CHANNELS)}


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _score_features(
    model: torch.nn.Module,
    feature_rows: np.ndarray,
    *,
    device: torch.device,
) -> np.ndarray:
    tensor = torch.as_tensor(feature_rows, dtype=torch.float32, device=device)
    with torch.no_grad():
        return model(tensor).reshape(-1).detach().cpu().numpy()


def _model_variant(
    *,
    key: str,
    group: str,
    examples: list,
    scores: np.ndarray,
    timing: dict[str, float | int],
    checkpoint_fingerprint: str,
    top_k: int,
    ablated_channel: str | None = None,
) -> dict[str, Any]:
    metrics = _method_result(key, examples, scores, timing, top_k=top_k)
    metrics.pop("method", None)
    return {
        "variant_key": key,
        "variant_group": group,
        "status": "completed",
        "scorer": "AdditiveAttentionScorer",
        "checkpoint_fingerprint": checkpoint_fingerprint,
        "diagnostic_only": ablated_channel is not None,
        "ablation": (
            {
                "type": "zeroed_channel",
                "removed_channels": [ablated_channel],
                "interpretation": "channel_sensitivity_under_distribution_shift",
            }
            if ablated_channel is not None
            else {"type": "none", "removed_channels": []}
        ),
        "metrics_scope": "window_ranking",
        "metrics": metrics,
    }


def evaluate_ltr_model_variants(
    *,
    manifest: str | Path,
    cache_dir: str | Path,
    checkpoint: str | Path,
    split: str,
    device: str,
    channels: Iterable[str] = FEATURE_CHANNELS,
    top_k: int = 5,
) -> dict[str, Any]:
    """Evaluate the full model and zero-one-channel sensitivity variants."""

    selected_channels = list(channels)
    unknown = sorted(set(selected_channels).difference(CHANNEL_INDEX))
    if unknown:
        raise ValueError(f"unknown feature channels: {', '.join(unknown)}")
    if len(selected_channels) != len(set(selected_channels)):
        raise ValueError("channels must not contain duplicates")

    target_device = _resolve_device(device)
    checkpoint_info = AdditiveAttentionScorer.preflight(checkpoint, device=target_device)
    records = load_training_manifest(manifest, split=split)
    examples = build_window_examples(cache_dir, records)
    if not examples:
        raise ValueError(f"manifest split {split!r} produced no evaluation windows")
    feature_rows = np.stack([example.feature for example in examples]).astype(np.float32)
    model, metadata = AdditiveAttentionScorer.load_checkpoint(
        checkpoint,
        device=target_device,
        expected_in_features=len(FEATURE_CHANNELS),
    )

    full_scores, full_timing = _measure_scores(
        lambda: _score_features(model, feature_rows, device=target_device),
        device=target_device,
    )
    variants = [
        _model_variant(
            key=FULL_LTR_VARIANT,
            group="full_ltr",
            examples=examples,
            scores=full_scores,
            timing=full_timing,
            checkpoint_fingerprint=checkpoint_info["fingerprint"],
            top_k=top_k,
        )
    ]
    for channel in selected_channels:
        ablated_rows = feature_rows.copy()
        ablated_rows[:, CHANNEL_INDEX[channel]] = 0.0
        ablated_scores, timing = _measure_scores(
            lambda rows=ablated_rows: _score_features(model, rows, device=target_device),
            device=target_device,
        )
        variants.append(
            _model_variant(
                key=f"ltr_without_{channel}",
                group="channel_ablation",
                examples=examples,
                scores=ablated_scores,
                timing=timing,
                checkpoint_fingerprint=checkpoint_info["fingerprint"],
                top_k=top_k,
                ablated_channel=channel,
            )
        )

    return {
        "manifest": str(manifest),
        "cache_dir": str(cache_dir),
        "checkpoint": str(checkpoint),
        "checkpoint_fingerprint": checkpoint_info["fingerprint"],
        "checkpoint_epoch": metadata.get("epoch"),
        "checkpoint_selection_ap": metadata.get("selection_ap", metadata.get("val_ap")),
        "dataset_fingerprint": metadata.get("dataset_fingerprint"),
        "split": split,
        "device": str(target_device),
        "video_count": len({example.video_id for example in examples}),
        "window_count": len(examples),
        "variants": variants,
    }


def _read_run_metadata(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"run metadata must be a JSON object: {source}")
    return payload


def classify_llm_run(path: str | Path) -> dict[str, Any]:
    """Classify one production artifact without inventing a fallback result."""

    source = Path(path)
    payload = _read_run_metadata(source)
    pipeline = payload.get("pipeline") or {}
    features = payload.get("features") or {}
    if not isinstance(pipeline, dict) or not isinstance(features, dict):
        raise ValueError(f"pipeline/features must be JSON objects: {source}")
    llm_run = pipeline.get("llm_run", payload.get("llm_run"))
    if not isinstance(llm_run, dict):
        raise ValueError(f"metadata does not contain llm_run: {source}")

    mode = pipeline.get("mode", features.get("mode"))
    enabled = llm_run.get("enabled") is True
    applied = llm_run.get("applied") is True
    if applied:
        variant_key = LLM_SUCCESS_VARIANT
        valid = enabled and mode == LLM_SUCCESS_VARIANT
        failure_reason = None if valid else "applied LLM metadata must use mode=ltr_llm_rerank"
        ranking_source = "ltr_plus_llm"
    elif enabled:
        variant_key = LLM_FAILURE_VARIANT
        fallback_reason = llm_run.get("fallback_reason")
        valid = mode == "ltr_required" and bool(fallback_reason)
        failure_reason = (
            None
            if valid
            else "LLM failure metadata requires mode=ltr_required and fallback_reason"
        )
        ranking_source = "ltr"
    else:
        raise ValueError(f"LLM was disabled in metadata: {source}")

    highlights = payload.get("highlights") or []
    if not isinstance(highlights, list):
        raise ValueError(f"highlights must be a JSON array: {source}")
    titled = [item for item in highlights if isinstance(item, dict) and item.get("title")]
    summarized = [item for item in highlights if isinstance(item, dict) and item.get("summary")]
    completeness = [
        float(item["completeness_score"])
        for item in highlights
        if isinstance(item, dict) and isinstance(item.get("completeness_score"), (int, float))
    ]
    checkpoint = pipeline.get("checkpoint") or payload.get("ltr_checkpoint_info") or {}
    if not isinstance(checkpoint, dict):
        checkpoint = {}
    return {
        "variant_key": variant_key,
        "variant_group": "llm_success" if applied else "llm_failure",
        "status": "completed" if valid else "failed",
        "failure_reason": failure_reason,
        "source_metadata": str(source),
        "pipeline_mode": mode,
        "ranking_source": ranking_source,
        "checkpoint_fingerprint": checkpoint.get("fingerprint"),
        "provider": llm_run.get("provider"),
        "model": llm_run.get("model"),
        "prompt_version": llm_run.get("prompt_version"),
        "cache_hit": bool(llm_run.get("cache_hit", False)),
        "assessed_count": int(llm_run.get("assessed_count", 0)),
        "fallback_reason": llm_run.get("fallback_reason"),
        "highlight_count": len(highlights),
        "title_count": len(titled),
        "summary_count": len(summarized),
        "mean_completeness": float(np.mean(completeness)) if completeness else None,
    }


def summarize_llm_runs(runs: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Keep successful reranks and provider failures in distinct report buckets."""

    rows = list(runs)
    summaries: dict[str, dict[str, Any]] = {}
    for key, group in (
        (LLM_SUCCESS_VARIANT, "llm_success"),
        (LLM_FAILURE_VARIANT, "llm_failure"),
    ):
        selected = [row for row in rows if row["variant_key"] == key]
        fallback_reasons = Counter(
            row["fallback_reason"] for row in selected if row.get("fallback_reason")
        )
        valid_count = sum(row["status"] == "completed" for row in selected)
        if not selected:
            status = "not_run"
        elif valid_count == len(selected):
            status = "completed"
        elif valid_count == 0:
            status = "failed"
        else:
            status = "partial_failure"
        summaries[key] = {
            "variant_key": key,
            "variant_group": group,
            "status": status,
            "run_count": len(selected),
            "valid_run_count": valid_count,
            "failed_metadata_count": sum(row["status"] == "failed" for row in selected),
            "ranking_source": "ltr_plus_llm" if key == LLM_SUCCESS_VARIANT else "ltr",
            "mean_highlight_count": (
                float(np.mean([row["highlight_count"] for row in selected])) if selected else None
            ),
            "mean_completeness": (
                float(
                    np.mean(
                        [row["mean_completeness"] for row in selected if row["mean_completeness"] is not None]
                    )
                )
                if any(row["mean_completeness"] is not None for row in selected)
                else None
            ),
            "fallback_reasons": dict(sorted(fallback_reasons.items())),
        }
    return summaries


def evaluate_variants(
    *,
    manifest: str | Path,
    cache_dir: str | Path,
    checkpoint: str | Path,
    split: str,
    device: str,
    channels: Iterable[str],
    run_metadata: Iterable[str | Path],
    top_k: int,
) -> dict[str, Any]:
    model_report = evaluate_ltr_model_variants(
        manifest=manifest,
        cache_dir=cache_dir,
        checkpoint=checkpoint,
        split=split,
        device=device,
        channels=channels,
        top_k=top_k,
    )
    llm_runs = [classify_llm_run(path) for path in run_metadata]
    return {
        "schema_version": "1.0",
        "evaluation_contract": {
            "hidden_fallback": False,
            "full_ltr": "checkpoint inference on all seven channels",
            "channel_ablation": "zero-one-channel sensitivity diagnostic using the full checkpoint",
            "llm_success": "production runs where llm_run.applied=true",
            "llm_failure": "production runs where LLM failed and ranking_source remains LTR",
        },
        "model_evaluation": model_report,
        "llm_variant_summary": summarize_llm_runs(llm_runs),
        "llm_runs": llm_runs,
        "metric_notes": {
            "model_variants": "Window-level ranking metrics from the selected manifest split.",
            "llm_variants": "Operational metadata only; quality metrics require aligned ground truth.",
            "channel_ablation": "Diagnostic zero-out, not a retrained ablation checkpoint.",
        },
    }


def _write_json(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "variant_group",
        "variant_key",
        "status",
        "checkpoint_fingerprint",
        "ablated_channel",
        "diagnostic_only",
        "average_precision",
        "kendall_tau",
        "spearman_rho",
        "window_f1_at_positive_count",
        "positive_hit_at_k",
        "source_metadata",
        "ranking_source",
        "fallback_reason",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for variant in result["model_evaluation"]["variants"]:
            metrics = variant["metrics"]
            writer.writerow(
                {
                    "variant_group": variant["variant_group"],
                    "variant_key": variant["variant_key"],
                    "status": variant["status"],
                    "checkpoint_fingerprint": variant["checkpoint_fingerprint"],
                    "ablated_channel": ",".join(variant["ablation"]["removed_channels"]),
                    "diagnostic_only": variant["diagnostic_only"],
                    **{field: metrics.get(field) for field in fields[6:11]},
                }
            )
        for run in result["llm_runs"]:
            writer.writerow({field: run.get(field) for field in fields})


def _format_metric(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.6f}"


def _write_markdown(path: Path, result: dict[str, Any]) -> None:
    model_report = result["model_evaluation"]
    lines = [
        "# LTR variant evaluation",
        "",
        f"Split: `{model_report['split']}`; device: `{model_report['device']}`; "
        f"videos: {model_report['video_count']}; windows: {model_report['window_count']}.",
        "",
        "## Full LTR and channel sensitivity",
        "",
        "| Group | Variant | AP | Kendall tau | Spearman rho | Window F1 | Hit@K |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for variant in model_report["variants"]:
        metrics = variant["metrics"]
        lines.append(
            f"| {variant['variant_group']} | {variant['variant_key']} | "
            f"{_format_metric(metrics['average_precision'])} | "
            f"{_format_metric(metrics['kendall_tau'])} | "
            f"{_format_metric(metrics['spearman_rho'])} | "
            f"{_format_metric(metrics['window_f1_at_positive_count'])} | "
            f"{_format_metric(metrics['positive_hit_at_k'])} |"
        )
    lines.extend(
        [
            "",
            "> `ltr_without_*` zeroes one channel while retaining the full checkpoint. "
            "This is a channel-sensitivity diagnostic, not retrained ablation.",
            "",
            "## LTR + LLM and LLM failure",
            "",
            "| Group | Variant | Status | Runs | Valid | Ranking source | Fallback reasons |",
            "|---|---|---|---:|---:|---|---|",
        ]
    )
    for summary in result["llm_variant_summary"].values():
        reasons = "; ".join(
            f"{reason} ({count})" for reason, count in summary["fallback_reasons"].items()
        )
        lines.append(
            f"| {summary['variant_group']} | {summary['variant_key']} | {summary['status']} | "
            f"{summary['run_count']} | {summary['valid_run_count']} | "
            f"{summary['ranking_source']} | {reasons or '-'} |"
        )
    lines.extend(
        [
            "",
            "> LLM rows are operational run reports. They are not assigned AP/correlation without "
            "aligned ground truth, and missing runs remain `not_run` instead of becoming random output.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/manifests/tvsum_smoke.jsonl")
    parser.add_argument("--cache-dir", default="data/features_cache")
    parser.add_argument("--checkpoint", default="data/models/ltr_scorer.pt")
    parser.add_argument("--split", default="val")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--channels", nargs="+", choices=FEATURE_CHANNELS, default=list(FEATURE_CHANNELS))
    parser.add_argument(
        "--run-metadata",
        action="append",
        default=[],
        help="Production metadata.json or run_agent summary; repeat for each LLM run.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-json", default="data/reports/ltr_variants.json")
    parser.add_argument("--output-csv", default="data/reports/ltr_variants.csv")
    parser.add_argument("--output-markdown", default="data/reports/ltr_variants.md")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    result = evaluate_variants(
        manifest=args.manifest,
        cache_dir=args.cache_dir,
        checkpoint=args.checkpoint,
        split=args.split,
        device=args.device,
        channels=args.channels,
        run_metadata=args.run_metadata,
        top_k=args.top_k,
    )
    _write_json(Path(args.output_json), result)
    _write_csv(Path(args.output_csv), result)
    _write_markdown(Path(args.output_markdown), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    main()
