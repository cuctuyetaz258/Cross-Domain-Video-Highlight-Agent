"""Join LTR/LLM candidate artifacts with graded custom annotations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from highlight_agent.models.train_offline import load_training_manifest


def candidate_importance(
    start: float,
    end: float,
    segments: list[list[float]],
) -> float:
    """Compute overlap-weighted importance without deriving pseudo boundaries."""

    if end <= start:
        raise ValueError("candidate end must be greater than start")
    weighted = 0.0
    covered = 0.0
    for segment_start, segment_end, score in segments:
        overlap = max(0.0, min(end, float(segment_end)) - max(start, float(segment_start)))
        weighted += overlap * float(score)
        covered += overlap
    if covered < (end - start) - 0.15:
        raise ValueError(
            f"annotation covers {covered:.3f}s of candidate duration {end - start:.3f}s"
        )
    return weighted / covered


def build_fusion_dataset(
    *,
    manifest_path: str | Path,
    metadata_paths: list[str | Path],
    output_path: str | Path,
) -> dict[str, Any]:
    records = {
        str(record["video_id"]): record for record in load_training_manifest(manifest_path)
    }
    rows: list[dict[str, Any]] = []
    seen_videos: set[str] = set()
    for metadata_path in metadata_paths:
        source = Path(metadata_path)
        payload = json.loads(source.read_text(encoding="utf-8"))
        video_id = str(payload.get("video_id", ""))
        if video_id not in records:
            raise ValueError(f"metadata video {video_id!r} is absent from custom manifest")
        if video_id in seen_videos:
            raise ValueError(f"multiple fusion metadata files supplied for video {video_id!r}")
        seen_videos.add(video_id)
        record = records[video_id]
        pipeline = payload.get("pipeline") or {}
        llm_run = pipeline.get("llm_run") or {}
        if not llm_run.get("applied"):
            raise ValueError(f"metadata has no successful LLM run: {source}")
        candidates = pipeline.get("fusion_candidates") or []
        if not candidates:
            raise ValueError(f"metadata has no fusion_candidates: {source}")
        checkpoint = pipeline.get("checkpoint") or {}
        for candidate in candidates:
            start = float(candidate["start_time"])
            end = float(candidate["end_time"])
            rows.append(
                {
                    "video_id": video_id,
                    "domain": record["domain"],
                    "candidate_id": candidate["candidate_id"],
                    "start_time": start,
                    "end_time": end,
                    "ltr_score": float(candidate["ltr_score"]),
                    "llm_score": float(candidate["llm_score"]),
                    "target_importance": candidate_importance(
                        start, end, record["importance_segments"]
                    ),
                    "split": record["split"],
                    "ltr_checkpoint_fingerprint": checkpoint.get("fingerprint"),
                    "llm_model": llm_run.get("model"),
                    "prompt_version": llm_run.get("prompt_version"),
                    "source_metadata": str(source),
                }
            )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(destination)
    return {
        "output": str(destination),
        "video_count": len(seen_videos),
        "candidate_count": len(rows),
        "splits": sorted({str(row["split"]) for row in rows}),
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run-metadata", action="append", required=True)
    parser.add_argument("--output", default="data/reports/fusion_candidates.jsonl")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report = build_fusion_dataset(
        manifest_path=args.manifest,
        metadata_paths=args.run_metadata,
        output_path=args.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
