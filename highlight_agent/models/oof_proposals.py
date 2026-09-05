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
from .proposal_protocol import (
    NESTED_OOF_VERSION,
    assert_lineage_allowed,
    example_digest,
    json_digest,
    split_contract,
)
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


def load_oof_proposal_cache(
    path: str | Path, *, allow_legacy_diagnostics: bool = False
) -> tuple[dict[str, list[TemporalProposal]], dict[str, Any]]:
    if not allow_legacy_diagnostics:
        raise ValueError("shared OOF v1 is unsafe for LTR training; use a nested OOF v2 cache")
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


def _checkpoint_record(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    model, metadata, _ = load_actionformer_checkpoint(source, device="cpu")
    return {
        "path": str(source),
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "config": json.loads(json.dumps(model.config.to_dict())),
        "data_lineage": metadata.get("data_lineage"),
        "content_fingerprints": metadata.get("content_fingerprints"),
    }


def build_nested_proposal_cache(
    *,
    outer_fold: int,
    train_examples: list[ActionFormerExample],
    val_video_ids: list[str],
    test_video_ids: list[str],
    generators: list[tuple[str | Path, list[ActionFormerExample]]],
    outer_checkpoint: str | Path,
    output_path: str | Path,
    device: str = "cpu",
) -> dict[str, Any]:
    """Cache only inner-held-out proposals, bound to one outer split and generator."""
    split = split_contract([x.video_id for x in train_examples], val_video_ids, test_video_ids)
    train_ids = set(split["train"])
    expected = {x.video_id: example_digest(x) for x in train_examples}
    outer = _checkpoint_record(outer_checkpoint)
    assert_lineage_allowed(outer["data_lineage"], train_ids)
    payload: dict[str, Any] = {
        "schema_version": NESTED_OOF_VERSION,
        "status": "running",
        "outer_fold": outer_fold,
        "outer_split": split,
        "outer_generator": outer,
        "generators": {},
        "videos": {},
        "started_at_unix": time.time(),
        "representation_policy": "outer_train_encoder_frozen_for_ltr; OOF applies to proposals only",
    }
    output = Path(output_path)
    _atomic_json(output, payload)
    try:
        for index, (checkpoint, targets) in enumerate(generators):
            target_ids = {x.video_id for x in targets}
            if not target_ids or len(target_ids) != len(targets) or not target_ids <= train_ids:
                raise ValueError("inner targets must be unique videos from outer train")
            record = _checkpoint_record(checkpoint)
            assert_lineage_allowed(record["data_lineage"], train_ids - target_ids)
            for key, digest in (record["content_fingerprints"] or {}).items():
                if expected.get(key) != digest:
                    raise ValueError(f"generator input content mismatch: {key}")
            model, _, _ = load_actionformer_checkpoint(checkpoint, device=device)
            generator_key = str(index)
            payload["generators"][generator_key] = record
            with torch.no_grad():
                for example in targets:
                    if example.video_id in payload["videos"]:
                        raise ValueError(f"duplicate nested OOF target: {example.video_id}")
                    if example_digest(example) != expected[example.video_id]:
                        raise ValueError("target content differs from outer train")
                    proposals = decode_proposals(
                        model(torch.from_numpy(example.features).unsqueeze(0).to(device)),
                        model.config,
                        video_durations=[example.duration],
                    )
                    payload["videos"][example.video_id] = {
                        "generator": generator_key,
                        "content_sha256": expected[example.video_id],
                        "duration": example.duration,
                        "proposals": [_proposal_payload(x) for x in proposals],
                    }
            _atomic_json(output, payload)
        if set(payload["videos"]) != train_ids:
            raise ValueError("nested OOF cache does not cover every outer-training video exactly once")
        payload["status"] = "complete"
        payload["completed_at_unix"] = time.time()
        payload["payload_sha256"] = json_digest(payload)
        _atomic_json(output, payload)
        return payload
    except BaseException as exc:
        payload.update(status="failed", error=str(exc), updated_at_unix=time.time())
        _atomic_json(output, payload)
        raise


def load_nested_proposal_cache(
    path: str | Path,
    *,
    train_examples: list[ActionFormerExample],
    val_video_ids: list[str],
    test_video_ids: list[str],
    outer_checkpoint_sha256: str | None = None,
) -> tuple[dict[str, list[TemporalProposal]], dict[str, Any]]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_version") != NESTED_OOF_VERSION:
        raise ValueError("requires nested OOF v2; shared OOF v1 is unsafe for training")
    if payload.get("status") != "complete":
        raise ValueError("nested OOF cache is incomplete")
    digest = payload.get("payload_sha256")
    if digest != json_digest({key: value for key, value in payload.items() if key != "payload_sha256"}):
        raise ValueError("nested OOF payload fingerprint mismatch")
    expected_split = split_contract([x.video_id for x in train_examples], val_video_ids, test_video_ids)
    if payload.get("outer_split") != expected_split:
        raise ValueError("nested OOF cache belongs to a different outer split")
    train_ids = set(expected_split["train"])
    expected_content = {x.video_id: example_digest(x) for x in train_examples}
    if set(payload.get("videos", {})) != train_ids:
        raise ValueError("nested OOF cache has missing or extra training videos")
    if outer_checkpoint_sha256 is not None and payload["outer_generator"]["sha256"] != outer_checkpoint_sha256:
        raise ValueError("cache was built for a different outer generator")
    records = {"outer": payload["outer_generator"], **payload["generators"]}
    for record in records.values():
        actual = _checkpoint_record(record["path"])
        if actual != record:
            raise ValueError("generator checkpoint fingerprint/metadata mismatch")
        assert_lineage_allowed(record["data_lineage"], train_ids)
        fingerprints = record.get("content_fingerprints")
        direct_ids = set(record["data_lineage"]["train_video_ids"] + record["data_lineage"]["selection_video_ids"])
        if not isinstance(fingerprints, dict) or set(fingerprints) != direct_ids:
            raise ValueError("generator is missing input content fingerprints")
        if any(expected_content.get(key) != value for key, value in fingerprints.items()):
            raise ValueError("generator training content fingerprint mismatch")
    predictions: dict[str, list[TemporalProposal]] = {}
    for video_id, record in payload["videos"].items():
        if record["content_sha256"] != expected_content[video_id]:
            raise ValueError(f"training content fingerprint mismatch: {video_id}")
        generator = payload["generators"][record["generator"]]
        assert_lineage_allowed(generator["data_lineage"], train_ids - {video_id})
        proposals = [TemporalProposal(**item) for item in record["proposals"]]
        for proposal in proposals:
            if not all(np.isfinite(x) for x in (proposal.start, proposal.end, proposal.confidence)):
                raise ValueError("non-finite cached proposal")
            if not (0 <= proposal.start < proposal.end <= record["duration"] and 0 <= proposal.confidence <= 1):
                raise ValueError("invalid cached proposal bounds/confidence")
            if not generator["config"]["min_duration_seconds"] <= proposal.duration <= generator["config"]["max_duration_seconds"]:
                raise ValueError("invalid cached proposal duration")
        predictions[video_id] = proposals
    return predictions, {
        "schema_version": NESTED_OOF_VERSION,
        "path": str(source.resolve()),
        "fingerprint": hashlib.sha256(source.read_bytes()).hexdigest(),
        "outer_fold": payload["outer_fold"],
        "outer_split": expected_split,
        "outer_generator": payload["outer_generator"],
        "generator_lineages": [record["data_lineage"] for record in payload["generators"].values()],
        "video_count": len(predictions),
        "representation_policy": payload["representation_policy"],
    }
