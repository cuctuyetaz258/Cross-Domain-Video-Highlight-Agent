"""Persist the expensive LTR analysis so LLM variants can reuse it."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from highlight_agent.backend import load_transcript
from highlight_agent.models.ltr_scorer import AdditiveAttentionScorer
from highlight_agent.schemas import HighlightCandidate, MediaWorkspace

from .state import AgentState

SNAPSHOT_SCHEMA_VERSION = "1.0"


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def save_analysis_snapshot(state: AgentState) -> tuple[Path, str]:
    """Write the reusable state produced by Observe/Plan/Analyze.

    API credentials and LLM settings are intentionally excluded.
    """

    workspace = state.get("workspace")
    transcript = state.get("transcript")
    candidates = state.get("candidates") or []
    checkpoint_info = state.get("ltr_checkpoint_info") or {}
    if workspace is None or transcript is None or not candidates:
        raise ValueError("analysis snapshot requires workspace, transcript, and LTR candidates")

    transcript_payload = transcript.model_dump(mode="json")
    identity_payload = {
        "video_id": workspace.video_id,
        "domain": state["domain"],
        "aspect_ratio": state.get("aspect_ratio", "9:16"),
        "transcript": transcript_payload,
        "checkpoint_fingerprint": checkpoint_info.get("fingerprint"),
        "feature_contract": state.get("features", {}).get("feature_contract", {}),
        "candidates": [item.model_dump(mode="json") for item in candidates],
    }
    analysis_id = hashlib.sha256(_canonical_json(identity_payload).encode("utf-8")).hexdigest()
    payload = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "analysis_id": analysis_id,
        "video_path": state["video_path"],
        "domain": state["domain"],
        "aspect_ratio": state.get("aspect_ratio", "9:16"),
        "workspace": workspace.model_dump(mode="json"),
        "transcript_sha256": hashlib.sha256(
            _canonical_json(transcript_payload).encode("utf-8")
        ).hexdigest(),
        "analysis_plan": state.get("analysis_plan", {}),
        "features": state.get("features", {}),
        "feature_path": state.get("feature_path"),
        "feature_timeline": state.get("feature_timeline", {}),
        "candidates": [item.model_dump(mode="json") for item in candidates],
        "ltr_checkpoint_info": checkpoint_info,
        "ltr_model_path": state.get("ltr_model_path"),
        "highlight_count": state.get("highlight_count", 3),
        "burn_subtitles": state.get("burn_subtitles", False),
    }
    snapshot_dir = workspace.transcript_path.parent / "analysis"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_dir / "ltr_analysis_snapshot.json"
    temporary_path = snapshot_path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary_path.replace(snapshot_path)
    return snapshot_path, analysis_id


def load_analysis_snapshot(
    snapshot_path: str | Path,
    *,
    ltr_model_path: str | Path | None = None,
) -> AgentState:
    """Restore analysis state and reject stale transcript/checkpoint combinations."""

    path = Path(snapshot_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"analysis snapshot does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported analysis snapshot schema: {payload.get('schema_version')!r}"
        )

    workspace = MediaWorkspace.model_validate(payload["workspace"])
    for required_path in (
        workspace.source_video_path,
        workspace.audio_path,
        workspace.transcript_path,
    ):
        if not required_path.is_file():
            raise FileNotFoundError(f"snapshot dependency does not exist: {required_path}")

    transcript = load_transcript(workspace.transcript_path)
    transcript_hash = hashlib.sha256(
        _canonical_json(transcript.model_dump(mode="json")).encode("utf-8")
    ).hexdigest()
    if transcript_hash != payload.get("transcript_sha256"):
        raise ValueError("transcript changed after the LTR analysis snapshot was created")

    checkpoint_path = str(ltr_model_path or payload.get("ltr_model_path") or "")
    checkpoint_info = AdditiveAttentionScorer.preflight(checkpoint_path)
    expected_fingerprint = payload.get("ltr_checkpoint_info", {}).get("fingerprint")
    if checkpoint_info.get("fingerprint") != expected_fingerprint:
        raise ValueError(
            "LTR checkpoint fingerprint differs from the checkpoint used by this snapshot"
        )

    return {
        "video_path": payload["video_path"],
        "domain": payload["domain"],
        "aspect_ratio": payload.get("aspect_ratio", "9:16"),
        "workspace": workspace,
        "transcript": transcript,
        "analysis_plan": payload.get("analysis_plan", {}),
        "features": payload.get("features", {}),
        "feature_path": payload.get("feature_path"),
        "feature_timeline": payload.get("feature_timeline", {}),
        "candidates": [
            HighlightCandidate.model_validate(item) for item in payload.get("candidates", [])
        ],
        "ltr_checkpoint_info": checkpoint_info,
        "ltr_model_path": checkpoint_path,
        "highlight_count": payload.get("highlight_count", 3),
        "burn_subtitles": payload.get("burn_subtitles", False),
        "analysis_snapshot_path": str(path),
        "analysis_id": payload["analysis_id"],
    }
