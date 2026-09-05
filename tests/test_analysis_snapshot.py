import json
from pathlib import Path

import pytest

from highlight_agent.agent import snapshot
from highlight_agent.schemas import (
    HighlightCandidate,
    MediaWorkspace,
    TranscriptDocument,
    TranscriptSegment,
)


def _state(tmp_path: Path) -> dict:
    workspace_dir = tmp_path / "output" / "video-01"
    workspace_dir.mkdir(parents=True)
    source_path = workspace_dir / "source_video.mp4"
    audio_path = workspace_dir / "audio.wav"
    transcript_path = workspace_dir / "transcript.json"
    source_path.write_bytes(b"video")
    audio_path.write_bytes(b"audio")
    transcript = TranscriptDocument(
        video_id="video-01",
        language="vi",
        source="whisper",
        duration=90,
        segments=[TranscriptSegment(id=0, start=0, end=90, text="Nội dung thử nghiệm.")],
    )
    transcript_path.write_text(transcript.model_dump_json(indent=2), encoding="utf-8")
    workspace = MediaWorkspace(
        video_id="video-01",
        source_type="local",
        original_input=str(source_path),
        source_video_path=source_path,
        audio_path=audio_path,
        transcript_path=transcript_path,
    )
    candidates = [
        HighlightCandidate(
            candidate_id=f"ltr-{index}",
            start_time=index * 30,
            end_time=(index + 1) * 30,
            score=1 - index * 0.1,
            reason="LTR",
        )
        for index in range(3)
    ]
    return {
        "video_path": str(source_path),
        "domain": "lecture",
        "aspect_ratio": "16:9",
        "workspace": workspace,
        "transcript": transcript,
        "features": {"mode": "ltr_required", "feature_contract": {"schema_version": "1.0"}},
        "feature_timeline": {},
        "candidates": candidates,
        "ltr_checkpoint_info": {"fingerprint": "checkpoint-a", "device": "cpu"},
        "ltr_model_path": "checkpoint.pt",
        "highlight_count": 3,
        "openai_api_key": "must-not-be-persisted",
    }


def test_snapshot_round_trip_excludes_credentials(tmp_path: Path, monkeypatch) -> None:
    state = _state(tmp_path)
    path, analysis_id = snapshot.save_analysis_snapshot(state)
    monkeypatch.setattr(
        snapshot.AdditiveAttentionScorer,
        "preflight",
        lambda path: {"fingerprint": "checkpoint-a", "device": "cpu"},
    )

    restored = snapshot.load_analysis_snapshot(path)

    assert restored["analysis_id"] == analysis_id
    assert restored["workspace"] == state["workspace"]
    assert restored["transcript"] == state["transcript"]
    assert restored["candidates"] == state["candidates"]
    assert restored["aspect_ratio"] == "16:9"
    assert "must-not-be-persisted" not in path.read_text(encoding="utf-8")


def test_snapshot_rejects_different_checkpoint(tmp_path: Path, monkeypatch) -> None:
    path, _ = snapshot.save_analysis_snapshot(_state(tmp_path))
    monkeypatch.setattr(
        snapshot.AdditiveAttentionScorer,
        "preflight",
        lambda path: {"fingerprint": "checkpoint-b", "device": "cpu"},
    )

    with pytest.raises(ValueError, match="fingerprint differs"):
        snapshot.load_analysis_snapshot(path)


def test_snapshot_rejects_modified_transcript(tmp_path: Path, monkeypatch) -> None:
    state = _state(tmp_path)
    path, _ = snapshot.save_analysis_snapshot(state)
    transcript_payload = json.loads(state["workspace"].transcript_path.read_text(encoding="utf-8"))
    transcript_payload["segments"][0]["text"] = "Transcript đã bị thay đổi."
    state["workspace"].transcript_path.write_text(
        json.dumps(transcript_payload, ensure_ascii=False), encoding="utf-8"
    )
    monkeypatch.setattr(
        snapshot.AdditiveAttentionScorer,
        "preflight",
        lambda path: {"fingerprint": "checkpoint-a", "device": "cpu"},
    )

    with pytest.raises(ValueError, match="transcript changed"):
        snapshot.load_analysis_snapshot(path)


def test_snapshot_restores_actionformer_checkpoint(tmp_path: Path, monkeypatch) -> None:
    state = _state(tmp_path)
    state.update(
        {
            "scorer_type": "actionformer-ltr",
            "actionformer_model_path": "actionformer.pt",
            "ltr_model_path": None,
        }
    )
    path, _ = snapshot.save_analysis_snapshot(state)
    monkeypatch.setattr(
        snapshot,
        "actionformer_checkpoint_info",
        lambda path: {"fingerprint": "checkpoint-a", "has_proposal_ltr": True},
    )

    restored = snapshot.load_analysis_snapshot(path)

    assert restored["scorer_type"] == "actionformer-ltr"
    assert restored["actionformer_model_path"] == "actionformer.pt"
    assert restored["ltr_model_path"] is None
