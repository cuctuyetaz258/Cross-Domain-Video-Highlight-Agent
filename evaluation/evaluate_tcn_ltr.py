"""Evaluate a V2 TCN-LTR checkpoint on one manifest split without LLM fusion."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from evaluation.evaluate_ltr import window_metrics
from highlight_agent.models.tcn_ltr_scorer import TemporalConvLTRScorer
from highlight_agent.models.train_offline import WindowExample, build_window_examples, load_training_manifest
from highlight_agent.models.train_tcn_ltr import score_examples


def evaluate(*, manifest: str | Path, cache_dir: str | Path, checkpoint: str | Path, split: str, device: str) -> dict[str, Any]:
    target = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device)
    records = load_training_manifest(manifest, split=split)
    examples = build_window_examples(cache_dir, records)
    model, metadata = TemporalConvLTRScorer.load_checkpoint(checkpoint, device=target)
    values = score_examples(model, examples, device=target)
    by_video: dict[str, list[WindowExample]] = defaultdict(list)
    for row in examples:
        by_video[row.video_id].append(row)
    per_video = []
    for video_id, rows in sorted(by_video.items()):
        per_video.append({"video_id": video_id, "domain": rows[0].domain, **window_metrics(rows, np.asarray([values[id(row)] for row in rows]), top_k=5)})
    metric_names = ("average_precision", "kendall_tau", "spearman_rho", "window_f1_at_positive_count", "positive_hit_at_k")
    macro = {
        name: float(np.mean([row[name] for row in per_video if row[name] is not None]))
        for name in metric_names
    }
    macro_sd = {
        name: float(np.std([row[name] for row in per_video if row[name] is not None], ddof=0))
        for name in metric_names
    }
    return {
        "schema_version": "1.0",
        "model_type": TemporalConvLTRScorer.model_type,
        "manifest": str(manifest),
        "split": split,
        "checkpoint": str(checkpoint),
        "device": target.type,
        "checkpoint_epoch": metadata.get("epoch"),
        "dataset_fingerprint": metadata.get("dataset_fingerprint"),
        "video_count": len(per_video),
        "window_count": len(examples),
        "macro_mean": macro,
        "macro_sd": macro_sd,
        "per_video": per_video,
        "metric_notes": "LTR-only evaluation; optional LLM rerank/fusion is disabled.",
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cache-dir", default="data/features_cache")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args(argv)
    result = evaluate(
        manifest=args.manifest,
        cache_dir=args.cache_dir,
        checkpoint=args.checkpoint,
        split=args.split,
        device=args.device,
    )
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("model_type", "split", "video_count", "macro_mean", "macro_sd")}, indent=2))


if __name__ == "__main__":
    main()
