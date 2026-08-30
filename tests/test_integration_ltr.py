"""Integration tests for the checkpoint-required LTR production path."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from highlight_agent.features.ltr_contract import (
    LTR_FEATURE_SCHEMA_VERSION,
    LTRPipelineError,
    feature_contract,
)
from highlight_agent.features.ltr_pipeline import LTRFeatureBundle
from highlight_agent.models.ltr_scorer import AdditiveAttentionScorer
from highlight_agent.schemas import HighlightCandidate, MediaWorkspace, TranscriptDocument, TranscriptSegment


def _workspace(tmp_path: Path) -> MediaWorkspace:
    return MediaWorkspace(
        video_id="test-video",
        source_type="local",
        original_input=str(tmp_path / "source.mp4"),
        source_video_path=tmp_path / "source.mp4",
        audio_path=tmp_path / "audio.wav",
        transcript_path=tmp_path / "transcript.json",
    )


def _checkpoint(tmp_path: Path) -> Path:
    path = tmp_path / "model.pt"
    AdditiveAttentionScorer().save(
        path,
        metadata={
            "schema_version": LTR_FEATURE_SCHEMA_VERSION,
            "feature_schema": feature_contract(),
            "L_ref": 40.0,
            "epoch": 2,
            "selection_ap": 0.8,
            "dataset_fingerprint": "fixture-dataset",
        },
    )
    return path


def test_preflight_rejects_missing_checkpoint_before_media() -> None:
    from highlight_agent.agent.nodes import preflight

    with pytest.raises(LTRPipelineError, match="LTR_CHECKPOINT_REQUIRED"):
        preflight({"video_path": "video.mp4", "domain": "lecture"})
    with pytest.raises(LTRPipelineError, match="LTR_CHECKPOINT_NOT_FOUND"):
        preflight(
            {
                "video_path": "video.mp4",
                "domain": "lecture",
                "ltr_model_path": "missing.pt",
            }
        )


def test_preflight_rejects_legacy_checkpoint_schema(tmp_path: Path) -> None:
    path = tmp_path / "legacy.pt"
    AdditiveAttentionScorer().save(path, metadata={"L_ref": 40.0})

    with pytest.raises(LTRPipelineError, match="LTR_CHECKPOINT_SCHEMA_MISMATCH"):
        AdditiveAttentionScorer.preflight(path)


def test_preflight_returns_contract_and_fingerprint(tmp_path: Path) -> None:
    info = AdditiveAttentionScorer.preflight(_checkpoint(tmp_path), device="cpu")

    assert info["device"] == "cpu"
    assert info["feature_contract"] == feature_contract()
    assert len(info["fingerprint"]) == 64


def test_analyze_runs_only_unified_ltr_path(tmp_path: Path, monkeypatch) -> None:
    from highlight_agent.agent import nodes

    workspace = _workspace(tmp_path)
    checkpoint_path = _checkpoint(tmp_path)
    transcript = TranscriptDocument(
        video_id="test-video",
        language="en",
        source="whisper",
        duration=120.0,
        segments=[TranscriptSegment(id=0, start=0, end=120, text="Complete transcript")],
    )
    acoustic = MagicMock()
    acoustic.duration = 120.0
    matrix = np.random.default_rng(42).random((7, 1200)).astype(np.float32)
    bundle = LTRFeatureBundle(
        matrix=matrix,
        acoustic=acoustic,
        acoustic_windows=[],
        interaction=None,
        metadata={
            "feature_contract": feature_contract(),
            "extractor": {"scene_status": "ok", "gesture_status": "ok"},
            "observations": {},
            "channel_stats": {},
        },
    )
    candidates = [
        HighlightCandidate(
            candidate_id=f"ltr_{index:02d}",
            start_time=float((index - 1) * 30),
            end_time=float(index * 30),
            score=1.0 / index,
            reason="LTR",
        )
        for index in range(1, 4)
    ]
    timeline = MagicMock()
    timeline.model_dump.return_value = {}
    checkpoint_info = AdditiveAttentionScorer.preflight(checkpoint_path, device="cpu")

    monkeypatch.setattr(nodes, "build_ltr_features", lambda **kwargs: bundle)
    monkeypatch.setattr(nodes, "build_feature_timeline", lambda **kwargs: timeline)
    monkeypatch.setattr(nodes, "save_feature_timeline", lambda *args: tmp_path / "features.json")
    monkeypatch.setattr(nodes, "extract_topk_nms", lambda *args, **kwargs: candidates)

    result = nodes.analyze(
        {
            "video_path": str(workspace.source_video_path),
            "domain": "lecture",
            "workspace": workspace,
            "transcript": transcript,
            "highlight_count": 3,
            "ltr_model_path": str(checkpoint_path),
            "ltr_checkpoint_info": checkpoint_info,
        }
    )

    assert result["features"]["mode"] == "ltr_required"
    assert result["candidates"] == candidates
    assert result["features"]["checkpoint"]["fingerprint"] == checkpoint_info["fingerprint"]
    assert not hasattr(nodes, "_naive_candidates")
    assert not hasattr(nodes, "_extract_visual_windows")


def test_analyze_fails_when_deterministic_pool_is_too_small(tmp_path: Path, monkeypatch) -> None:
    from highlight_agent.agent import nodes

    workspace = _workspace(tmp_path)
    checkpoint_path = _checkpoint(tmp_path)
    transcript = TranscriptDocument(
        video_id="test-video",
        language="en",
        source="whisper",
        duration=120.0,
        segments=[TranscriptSegment(id=0, start=0, end=120, text="Transcript")],
    )
    acoustic = MagicMock()
    acoustic.duration = 120.0
    bundle = LTRFeatureBundle(
        np.zeros((7, 1200), dtype=np.float32),
        acoustic,
        [],
        None,
        {
            "feature_contract": feature_contract(),
            "extractor": {},
            "observations": {},
            "channel_stats": {},
        },
    )
    timeline = MagicMock()
    timeline.model_dump.return_value = {}
    monkeypatch.setattr(nodes, "build_ltr_features", lambda **kwargs: bundle)
    monkeypatch.setattr(nodes, "build_feature_timeline", lambda **kwargs: timeline)
    monkeypatch.setattr(nodes, "save_feature_timeline", lambda *args: tmp_path / "features.json")
    monkeypatch.setattr(nodes, "extract_topk_nms", lambda *args, **kwargs: [])

    with pytest.raises(LTRPipelineError, match="LTR_NOT_ENOUGH_CANDIDATES"):
        nodes.analyze(
            {
                "video_path": str(workspace.source_video_path),
                "domain": "lecture",
                "workspace": workspace,
                "transcript": transcript,
                "highlight_count": 3,
                "ltr_model_path": str(checkpoint_path),
            }
        )


def test_cli_defaults_to_required_checkpoint_and_removes_visual_flags() -> None:
    from scripts.run_agent import parse_args

    args = parse_args(["video.mp4", "--domain", "lecture"])
    assert args.ltr_model_path == "data/models/ltr_target_lecture_podcast.pt"
    ranged = parse_args(
        ["video.mp4", "--domain", "podcast", "--min-speaker-count", "1", "--max-speaker-count", "3"]
    )
    assert (ranged.min_speaker_count, ranged.max_speaker_count) == (1, 3)
    with pytest.raises(SystemExit):
        parse_args(
            [
                "video.mp4",
                "--domain",
                "lecture",
                "--visual-method",
                "pixel_diff",
            ]
        )
    with pytest.raises(SystemExit):
        parse_args(
            ["video.mp4", "--domain", "podcast", "--known-speaker-count", "2", "--min-speaker-count", "1"]
        )


def test_cli_serializes_ltr_failure_and_exits_nonzero(monkeypatch, capsys) -> None:
    from scripts import run_agent

    def fail() -> None:
        raise LTRPipelineError("LTR_CHECKPOINT_NOT_FOUND", "missing fixture")

    monkeypatch.setattr(run_agent, "main", fail)

    with pytest.raises(SystemExit) as exit_info:
        run_agent.cli()

    assert exit_info.value.code == 2
    assert capsys.readouterr().err.strip() == (
        '{"error": {"code": "LTR_CHECKPOINT_NOT_FOUND", '
        '"message": "missing fixture"}}'
    )
