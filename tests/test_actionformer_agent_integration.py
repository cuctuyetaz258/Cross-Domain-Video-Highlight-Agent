from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from highlight_agent.agent import nodes
from highlight_agent.models.actionformer import (
    ActionFormerConfig,
    ActionFormerHighlightModel,
    TemporalProposal,
    save_actionformer_checkpoint,
)
from highlight_agent.models.proposal_ltr import ProposalLTRScorer
from highlight_agent.schemas import MediaWorkspace, TranscriptDocument, TranscriptSegment


def test_analyze_actionformer_ltr_produces_variable_candidates(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = ActionFormerConfig(
        d_model=8,
        num_heads=2,
        attention_window=16,
        pyramid_levels=2,
        head_depth=1,
        dropout=0.0,
        regression_ranges_seconds=((0.0, 45.0), (30.0, float("inf"))),
    )
    model = ActionFormerHighlightModel(config)
    torch.nn.init.zeros_(model.head.classifier.weight)
    torch.nn.init.constant_(model.head.classifier.bias, 10.0)
    torch.nn.init.zeros_(model.head.regressor.weight)
    torch.nn.init.constant_(model.head.regressor.bias, 40.0)
    proposal_ltr = ProposalLTRScorer(config.d_model)
    checkpoint = tmp_path / "actionformer_ltr.pt"
    save_actionformer_checkpoint(
        checkpoint,
        model,
        metadata={
            "feature_schema_version": "1.1",
            "channel_order": [
                "rms",
                "pitch",
                "silence",
                "text_score",
                "scene_change",
                "gesture",
                "turn_rate",
            ],
            "dataset_fingerprint": "dataset",
            "split_fingerprint": "split",
            "normalization_policy_version": "duration_30_90_v1",
        },
        proposal_ltr_state_dict=proposal_ltr.state_dict(),
    )
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    source = media_dir / "source.mp4"
    audio = media_dir / "audio.wav"
    transcript_path = media_dir / "transcript.json"
    source.write_bytes(b"video")
    audio.write_bytes(b"audio")
    transcript = TranscriptDocument(
        video_id="video",
        language="vi",
        source="whisper",
        duration=120.0,
        segments=[TranscriptSegment(id=0, start=0.0, end=120.0, text="fixture")],
    )
    transcript_path.write_text(transcript.model_dump_json(), encoding="utf-8")
    workspace = MediaWorkspace(
        video_id="video",
        source_type="local",
        original_input=str(source),
        source_video_path=source,
        audio_path=audio,
        transcript_path=transcript_path,
    )
    bundle = SimpleNamespace(
        matrix=np.random.default_rng(42).random((7, 1200), dtype=np.float32),
        interaction=None,
        acoustic=SimpleNamespace(duration=120.0),
        acoustic_windows=[],
        metadata={
            "feature_contract": {"schema_version": "1.1"},
            "extractor": {},
            "observations": {},
            "channel_stats": {},
        },
    )
    monkeypatch.setattr(nodes, "build_ltr_features", lambda **kwargs: bundle)
    monkeypatch.setattr(
        nodes,
        "decode_proposals",
        lambda outputs, config, video_durations: [
            TemporalProposal(float(start), float(start + 40), 0.9, 0, index)
            for index, start in enumerate((0, 10, 20, 40, 60, 80))
        ],
    )
    monkeypatch.setattr(
        nodes,
        "build_feature_timeline",
        lambda **kwargs: SimpleNamespace(model_dump=lambda mode: {}),
    )
    monkeypatch.setattr(
        nodes,
        "save_feature_timeline",
        lambda timeline, path: Path(path),
    )

    result = nodes.analyze(
        {
            "video_path": str(source),
            "domain": "lecture",
            "workspace": workspace,
            "transcript": transcript,
            "scorer_type": "actionformer-ltr",
            "actionformer_model_path": str(checkpoint),
            "highlight_count": 3,
            "candidate_pool_size": 5,
            "llm_provider": "disabled",
        }
    )

    assert result["features"]["mode"] == "actionformer_ltr"
    assert len(result["candidates"]) == 5
    assert all(
        30.0 <= item.end_time - item.start_time <= 90.0
        for item in result["candidates"]
    )
    assert all("ltr_score_raw" in item.signals for item in result["candidates"])
