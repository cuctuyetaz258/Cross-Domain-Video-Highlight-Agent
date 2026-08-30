"""Fit percentile-rank LTR–LLM alpha on candidate-level validation data."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from highlight_agent.llm.fusion import FusionCalibrator, fuse_ranked_scores


def _ndcg(relevances: np.ndarray, scores: np.ndarray, k: int) -> float:
    if relevances.size == 0:
        return 0.0
    limit = min(k, relevances.size)
    order = np.argsort(-scores, kind="stable")[:limit]
    ideal = np.argsort(-relevances, kind="stable")[:limit]
    discounts = np.log2(np.arange(2, limit + 2, dtype=np.float64))
    dcg = float(np.sum((np.power(2.0, relevances[order]) - 1.0) / discounts))
    ideal_dcg = float(np.sum((np.power(2.0, relevances[ideal]) - 1.0) / discounts))
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def load_fusion_records(
    paths: str | Path | Iterable[str | Path], *, split: str = "val"
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    selected_paths = [paths] if isinstance(paths, (str, Path)) else list(paths)
    for path in selected_paths:
        with Path(path).open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                required = {
                    "video_id",
                    "domain",
                    "candidate_id",
                    "ltr_score",
                    "llm_score",
                    "target_importance",
                    "split",
                }
                missing = sorted(required.difference(record))
                if missing:
                    raise ValueError(
                        f"{path}:{line_number} missing fields: {', '.join(missing)}"
                    )
                if record["split"] != split:
                    continue
                if record["domain"] not in {"lecture", "podcast"}:
                    raise ValueError(f"{path}:{line_number} has unsupported target domain")
                for field in ("ltr_score", "llm_score", "target_importance"):
                    value = float(record[field])
                    if not np.isfinite(value):
                        raise ValueError(f"{path}:{line_number} {field} must be finite")
                    record[field] = value
                records.append(record)
    if not records:
        raise ValueError(f"fusion datasets contain no records for split {split!r}")
    return records


def macro_domain_ndcg(
    records: Iterable[dict[str, Any]],
    *,
    alpha: float,
    k: int = 3,
) -> tuple[float, dict[str, float], dict[str, float]]:
    """Average per video first, then macro-average lecture and podcast."""

    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_video[str(record["video_id"])].append(record)
    video_metrics: dict[str, float] = {}
    video_domains: dict[str, str] = {}
    for video_id, candidates in sorted(by_video.items()):
        domains = {str(candidate["domain"]) for candidate in candidates}
        if len(domains) != 1:
            raise ValueError(f"video {video_id!r} has multiple domains")
        final_scores = fuse_ranked_scores(
            [candidate["ltr_score"] for candidate in candidates],
            [candidate["llm_score"] for candidate in candidates],
            alpha=alpha,
        )
        relevances = np.asarray(
            [candidate["target_importance"] for candidate in candidates], dtype=np.float64
        )
        video_metrics[video_id] = _ndcg(relevances, final_scores, k)
        video_domains[video_id] = next(iter(domains))

    domain_metrics = {
        domain: float(
            np.mean(
                [score for video_id, score in video_metrics.items() if video_domains[video_id] == domain]
            )
        )
        for domain in sorted(set(video_domains.values()))
    }
    return float(np.mean(list(domain_metrics.values()))), domain_metrics, video_metrics


def fit_global_alpha(
    records: list[dict[str, Any]],
    *,
    step: float = 0.05,
    k: int = 3,
) -> dict[str, Any]:
    if not 0 < step <= 1:
        raise ValueError("alpha step must be within (0, 1]")
    domains = {str(record["domain"]) for record in records}
    if domains != {"lecture", "podcast"}:
        raise ValueError("fusion validation data must contain lecture and podcast")
    candidates = np.arange(0.0, 1.0 + step / 2.0, step)
    rows: list[dict[str, Any]] = []
    for raw_alpha in candidates:
        alpha = min(1.0, round(float(raw_alpha), 10))
        macro, by_domain, by_video = macro_domain_ndcg(records, alpha=alpha, k=k)
        rows.append(
            {
                "alpha": alpha,
                "macro_ndcg": macro,
                "ndcg_by_domain": by_domain,
                "ndcg_by_video": by_video,
            }
        )
    best = max(rows, key=lambda row: (row["macro_ndcg"], row["alpha"]))
    return {"best": best, "grid": rows, "k": k, "step": step}


def _fingerprint(records: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: (str(item["video_id"]), str(item["candidate_id"]))):
        digest.update(json.dumps(record, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def _single_metadata(records: list[dict[str, Any]], field: str) -> str | None:
    values = {str(record[field]) for record in records if record.get(field)}
    if len(values) > 1:
        raise ValueError(f"fusion records contain multiple {field} values")
    return next(iter(values)) if values else None


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def train_fusion(
    *,
    input_path: str | Path | Iterable[str | Path],
    output_path: str | Path,
    report_path: str | Path,
    split: str = "val",
    step: float = 0.05,
    k: int = 3,
    bind_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    records = load_fusion_records(input_path, split=split)
    result = fit_global_alpha(records, step=step, k=k)
    best = result["best"]
    training_checkpoint_fingerprints = tuple(
        sorted(
            {
                str(record["ltr_checkpoint_fingerprint"])
                for record in records
                if record.get("ltr_checkpoint_fingerprint")
            }
        )
    )
    bound_fingerprint = (
        _file_sha256(bind_checkpoint)
        if bind_checkpoint
        else training_checkpoint_fingerprints[0]
        if len(training_checkpoint_fingerprints) == 1
        else None
    )
    calibrator = FusionCalibrator(
        alpha=float(best["alpha"]),
        selection_metric=f"macro_ndcg@{k}",
        selection_score=float(best["macro_ndcg"]),
        ltr_checkpoint_fingerprint=bound_fingerprint,
        llm_model=_single_metadata(records, "llm_model"),
        prompt_version=_single_metadata(records, "prompt_version"),
        training_dataset_fingerprint=_fingerprint(records),
        training_checkpoint_fingerprints=training_checkpoint_fingerprints,
    )
    output = Path(output_path)
    report = Path(report_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(calibrator.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"calibrator": calibrator.to_dict(), "report": result}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Candidate-level fusion JSONL; repeat to fit global alpha across folds.",
    )
    parser.add_argument("--output", default="data/models/fusion_calibrator.json")
    parser.add_argument("--report", default="data/reports/fusion_grid_search.json")
    parser.add_argument("--split", default="val")
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument(
        "--bind-checkpoint",
        default=None,
        help="Release LTR checkpoint whose SHA-256 must match this calibrator at runtime.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result = train_fusion(
        input_path=args.input,
        output_path=args.output,
        report_path=args.report,
        split=args.split,
        step=args.step,
        k=args.k,
        bind_checkpoint=args.bind_checkpoint,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
