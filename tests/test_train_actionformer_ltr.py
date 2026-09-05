import json

import numpy as np
import pytest

from highlight_agent.models.actionformer import (
    ActionFormerConfig,
    ActionFormerHighlightModel,
    TemporalProposal,
)
from highlight_agent.models.train_actionformer_ltr import (
    ActionFormerExample,
    proposal_utility,
    train_actionformer_localization,
    train_proposal_ltr,
)


def _example(video_id: str, offset: float = 0.0) -> ActionFormerExample:
    return ActionFormerExample(
        video_id=video_id,
        domain="lecture",
        duration=60.0,
        features=np.random.default_rng(42).random((7, 600), dtype=np.float32),
        boundaries=np.asarray([[15.0 + offset, 45.0 + offset]], dtype=np.float32),
        importance=np.asarray([[0.0, 30.0, 1.0], [30.0, 60.0, 5.0]], dtype=np.float32),
    )


def test_training_persists_incremental_report_artifacts(tmp_path) -> None:
    config = ActionFormerConfig(
        d_model=8,
        num_heads=2,
        attention_window=16,
        pyramid_levels=2,
        head_depth=1,
        dropout=0.0,
        regression_ranges_seconds=((0.0, 45.0), (30.0, float("inf"))),
    )
    _, report = train_actionformer_localization(
        train_examples=[_example("train")],
        val_examples=[_example("val")],
        output_path=tmp_path / "best.pt",
        last_output_path=tmp_path / "last.pt",
        log_path=tmp_path / "log.json",
        history_csv_path=tmp_path / "history.csv",
        curves_path=tmp_path / "curves.svg",
        config=config,
        max_epochs=1,
        patience=1,
        device="cpu",
    )

    saved = json.loads((tmp_path / "log.json").read_text(encoding="utf-8"))
    assert report["status"] == "complete"
    assert saved["status"] == "complete"
    assert len(saved["epochs"]) == 1
    assert (tmp_path / "best.pt").is_file()
    assert (tmp_path / "last.pt").is_file()
    assert (tmp_path / "history.csv").is_file()
    assert (tmp_path / "curves.svg").is_file()


def test_localization_can_initialize_from_a_compatible_checkpoint(tmp_path) -> None:
    config = ActionFormerConfig(
        d_model=8,
        num_heads=2,
        attention_window=16,
        pyramid_levels=2,
        head_depth=1,
        dropout=0.0,
        regression_ranges_seconds=((0.0, 45.0), (30.0, float("inf"))),
    )
    source = tmp_path / "pretrained.pt"
    train_actionformer_localization(
        train_examples=[_example("source_train")],
        val_examples=[_example("source_val")],
        output_path=source,
        last_output_path=tmp_path / "pretrained_last.pt",
        log_path=tmp_path / "pretrained_log.json",
        history_csv_path=tmp_path / "pretrained_history.csv",
        curves_path=tmp_path / "pretrained_curves.svg",
        config=config,
        max_epochs=1,
        patience=1,
        device="cpu",
    )

    _, report = train_actionformer_localization(
        train_examples=[_example("target_train")],
        val_examples=[_example("target_val")],
        output_path=tmp_path / "finetuned.pt",
        last_output_path=tmp_path / "finetuned_last.pt",
        log_path=tmp_path / "finetuned_log.json",
        history_csv_path=tmp_path / "finetuned_history.csv",
        curves_path=tmp_path / "finetuned_curves.svg",
        config=config,
        init_checkpoint_path=source,
        max_epochs=1,
        patience=1,
        device="cpu",
    )

    initialization = report["initialization"]
    assert initialization["checkpoint_path"] == str(source.resolve())
    assert len(initialization["checkpoint_sha256"]) == 64


def test_localization_can_resume_from_last_checkpoint(tmp_path) -> None:
    config = ActionFormerConfig(
        d_model=8,
        num_heads=2,
        attention_window=16,
        pyramid_levels=2,
        head_depth=1,
        dropout=0.0,
        regression_ranges_seconds=((0.0, 45.0), (30.0, float("inf"))),
    )
    common = {
        "train_examples": [_example("train")],
        "val_examples": [_example("val")],
        "config": config,
        "output_path": tmp_path / "best.pt",
        "last_output_path": tmp_path / "last.pt",
        "log_path": tmp_path / "log.json",
        "history_csv_path": tmp_path / "history.csv",
        "curves_path": tmp_path / "curves.svg",
        "patience": 4,
        "device": "cpu",
    }
    train_actionformer_localization(**common, max_epochs=1)
    _, report = train_actionformer_localization(**common, max_epochs=2, resume_checkpoint=tmp_path / "last.pt")
    assert [row["epoch"] for row in report["epochs"]] == [1, 2]


def test_proposal_ltr_training_persists_combined_checkpoint_and_log(tmp_path) -> None:
    config = ActionFormerConfig(
        d_model=8,
        num_heads=2,
        attention_window=16,
        pyramid_levels=2,
        head_depth=1,
        dropout=0.0,
        regression_ranges_seconds=((0.0, 45.0), (30.0, float("inf"))),
    )
    metadata = {
        "feature_schema_version": "1.1",
        "channel_order": ["rms", "pitch", "silence", "text_score", "scene_change", "gesture", "turn_rate"],
        "dataset_fingerprint": "dataset",
        "split_fingerprint": "split",
        "normalization_policy_version": "duration_30_90_v1",
    }

    _, report = train_proposal_ltr(
        actionformer=ActionFormerHighlightModel(config),
        checkpoint_metadata=metadata,
        train_examples=[_example("train")],
        val_examples=[_example("val")],
        output_path=tmp_path / "combined.pt",
        last_output_path=tmp_path / "combined_last.pt",
        log_path=tmp_path / "ltr_log.json",
        history_csv_path=tmp_path / "ltr_history.csv",
        curves_path=tmp_path / "ltr_curves.svg",
        max_epochs=1,
        patience=1,
        device="cpu",
    )

    assert report["status"] == "complete"
    assert report["best_epoch"] == 1
    assert (tmp_path / "combined.pt").is_file()
    assert json.loads((tmp_path / "ltr_log.json").read_text())["stage"] == "proposal_ltr"


def test_proposal_utility_v2_rewards_boundary_alignment() -> None:
    example = _example("utility")
    aligned = TemporalProposal(15.0, 45.0, 0.5, 0, 1)
    oversized = TemporalProposal(0.0, 60.0, 0.5, 0, 2)

    assert proposal_utility(example, aligned) > proposal_utility(example, oversized)


def test_proposal_ltr_rejects_unverified_external_proposal_cache(tmp_path) -> None:
    config = ActionFormerConfig(
        d_model=8,
        num_heads=2,
        attention_window=16,
        pyramid_levels=2,
        head_depth=1,
        dropout=0.0,
        regression_ranges_seconds=((0.0, 45.0), (30.0, float("inf"))),
    )
    metadata = {
        "feature_schema_version": "1.1",
        "channel_order": ["rms", "pitch", "silence", "text_score", "scene_change", "gesture", "turn_rate"],
        "dataset_fingerprint": "dataset",
        "split_fingerprint": "split",
        "normalization_policy_version": "duration_30_90_v1",
    }
    cached = {
        "train": [TemporalProposal(5.0, 35.0, 0.9, 0, 1)],
        "val": [TemporalProposal(10.0, 40.0, 0.8, 0, 2)],
    }

    with pytest.raises(ValueError, match="unverified"):
        train_proposal_ltr(
            actionformer=ActionFormerHighlightModel(config),
            checkpoint_metadata=metadata,
            train_examples=[_example("train")],
            val_examples=[_example("val")],
            output_path=tmp_path / "combined.pt",
            last_output_path=tmp_path / "combined_last.pt",
            log_path=tmp_path / "ltr_log.json",
            history_csv_path=tmp_path / "ltr_history.csv",
            curves_path=tmp_path / "ltr_curves.svg",
            max_epochs=1,
            patience=1,
            device="cpu",
            predicted_proposals_by_video=cached,
            proposal_cache_metadata={"fingerprint": "cache"},
        )
