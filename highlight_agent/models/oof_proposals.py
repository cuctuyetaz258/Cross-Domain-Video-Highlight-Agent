from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .actionformer import TemporalProposal, decode_proposals, load_actionformer_checkpoint
from .train_actionformer_ltr import ActionFormerExample, load_actionformer_manifest

OOF_PROPOSAL_CACHE_VERSION = "actionformer_oof_proposals_v1"


def _examples_fingerprint(examples: list[ActionFormerExample]) -> str:
    payload = [
        {
            "video_id": item.video_id,
            "domain": item.domain,
            "duration": item.duration,
            "boundaries": item.boundaries.tolist(),
            "shape": list(item.features.shape),
            "importance_shape": list(item.importance.shape),
        }
        for item in sorted(examples, key=lambda value: value.video_id)
    ]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _proposal_payload(proposal: TemporalProposal) -> dict[str, float | int]:
    return {
        "start": proposal.start,
        "end": proposal.end,
        "confidence": proposal.confidence,
        "level": proposal.level,
        "center_index": proposal.center_index,
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _write_stats_csv(path: Path, videos: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["video_id", "fold", "domain", "duration", "proposal_count"],
        )
        writer.writeheader()
        for video_id, record in sorted(videos.items()):
            writer.writerow(
                {
                    "video_id": video_id,
                    "fold": record["fold"],
                    "domain": record["domain"],
                    "duration": record["duration"],
                    "proposal_count": len(record["proposals"]),
                }
            )
    temporary.replace(path)


def build_oof_proposal_cache(
    *,
    fold_inputs: list[tuple[int, str | Path, str | Path]],
    output_path: str | Path,
    stats_csv_path: str | Path,
    project_root: str | Path = ".",
    device: str = "cpu",
) -> dict[str, Any]:
    """Build predicted proposals only for held-out test videos of each fold."""
    output = Path(output_path)
    stats_csv = Path(stats_csv_path)
    started = time.time()
    report: dict[str, Any] = {
        "schema_version": OOF_PROPOSAL_CACHE_VERSION,
        "status": "running",
        "started_at_unix": started,
        "updated_at_unix": started,
        "device": device,
        "folds": [],
        "videos": {},
    }
    _atomic_json(output, report)

    for fold, manifest_path, checkpoint_path in fold_inputs:
        train_examples = load_actionformer_manifest(manifest_path, split="train", project_root=project_root)
        val_examples = load_actionformer_manifest(manifest_path, split="val", project_root=project_root)
        test_examples = load_actionformer_manifest(manifest_path, split="test", project_root=project_root)
        train_ids = {item.video_id for item in train_examples}
        held_out_ids = {item.video_id for item in test_examples}
        if train_ids & held_out_ids:
            raise ValueError(f"fold {fold} contains train/test video leakage")

        model, metadata, _ = load_actionformer_checkpoint(checkpoint_path, device=device)
        train_fingerprint = _examples_fingerprint(train_examples)
        val_fingerprint = _examples_fingerprint(val_examples)
        split_fingerprint = hashlib.sha256(
            f"{train_fingerprint}:{val_fingerprint}".encode()
        ).hexdigest()
        if metadata.get("dataset_fingerprint") != train_fingerprint:
            raise ValueError(f"fold {fold} checkpoint does not match the manifest training split")
        if metadata.get("validation_fingerprint") != val_fingerprint:
            raise ValueError(f"fold {fold} checkpoint does not match the manifest validation split")
        if metadata.get("split_fingerprint") != split_fingerprint:
            raise ValueError(f"fold {fold} checkpoint split fingerprint is invalid")

        checkpoint_source = Path(checkpoint_path)
        checkpoint_fingerprint = hashlib.sha256(checkpoint_source.read_bytes()).hexdigest()
        proposal_counts: list[int] = []
        model.eval()
        with torch.no_grad():
            for example in test_examples:
                if example.video_id in report["videos"]:
                    raise ValueError(f"duplicate OOF video: {example.video_id}")
                features = torch.from_numpy(example.features).unsqueeze(0).to(device)
                proposals = decode_proposals(
                    model(features),
                    model.config,
                    video_durations=[example.duration],
                )
                proposal_counts.append(len(proposals))
                report["videos"][example.video_id] = {
                    "fold": fold,
                    "domain": example.domain,
                    "duration": example.duration,
                    "checkpoint_fingerprint": checkpoint_fingerprint,
                    "proposals": [_proposal_payload(item) for item in proposals],
                }

        report["folds"].append(
            {
                "fold": fold,
                "manifest": str(manifest_path),
                "checkpoint": str(checkpoint_path),
                "checkpoint_fingerprint": checkpoint_fingerprint,
                "train_video_ids": sorted(train_ids),
                "test_video_ids": sorted(held_out_ids),
                "video_count": len(test_examples),
                "proposal_count": sum(proposal_counts),
                "mean_proposals_per_video": float(np.mean(proposal_counts)) if proposal_counts else 0.0,
            }
        )
        report["updated_at_unix"] = time.time()
        _atomic_json(output, report)
        _write_stats_csv(stats_csv, report["videos"])

    report["status"] = "complete"
    report["completed_at_unix"] = time.time()
    report["elapsed_seconds"] = report["completed_at_unix"] - started
    report["summary"] = {
        "fold_count": len(report["folds"]),
        "video_count": len(report["videos"]),
        "proposal_count": sum(len(item["proposals"]) for item in report["videos"].values()),
    }
    _atomic_json(output, report)
    _write_stats_csv(stats_csv, report["videos"])
    return report


def load_oof_proposal_cache(path: str | Path) -> tuple[dict[str, list[TemporalProposal]], dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != OOF_PROPOSAL_CACHE_VERSION:
        raise ValueError("unsupported OOF proposal cache schema")
    if payload.get("status") != "complete":
        raise ValueError("OOF proposal cache is incomplete")
    videos = payload.get("videos")
    if not isinstance(videos, dict):
        raise ValueError("OOF proposal cache is missing videos")
    proposals_by_video: dict[str, list[TemporalProposal]] = {}
    for video_id, record in videos.items():
        proposals_by_video[str(video_id)] = [
            TemporalProposal(
                start=float(item["start"]),
                end=float(item["end"]),
                confidence=float(item["confidence"]),
                level=int(item["level"]),
                center_index=int(item["center_index"]),
            )
            for item in record["proposals"]
        ]
    metadata = {
        "schema_version": payload["schema_version"],
        "path": str(Path(path).resolve()),
        "fingerprint": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "fold_count": len(payload.get("folds", [])),
        "video_count": len(proposals_by_video),
    }
    return proposals_by_video, metadata
