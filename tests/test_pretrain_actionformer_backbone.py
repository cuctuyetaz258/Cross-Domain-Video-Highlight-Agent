"""Regression tests for benchmark-pretraining label alignment and recovery."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from highlight_agent.models.actionformer import (
    ActionFormerConfig,
    ActionFormerHighlightModel,
    save_actionformer_checkpoint,
)


def _module():
    path = Path(__file__).parents[1] / "scripts" / "pretrain_actionformer_backbone.py"
    spec = importlib.util.spec_from_file_location("pretrain_actionformer_backbone", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_timestamp_alignment_uses_exact_stem_support() -> None:
    runner = _module()
    config = ActionFormerConfig(
        d_model=8,
        num_heads=2,
        attention_window=8,
        pyramid_levels=2,
        regression_ranges_seconds=((0.0, 45.0), (30.0, float("inf"))),
    )
    row = {"video_id": "sample", "source": "tvsum", "fps": 10.0, "frame_scores": list(range(10))}
    matrix = np.zeros((7, 10), dtype=np.float32)
    labels = runner._level_zero_labels(row, matrix, {"tvsum": (0.0, 9.0)}, config)
    # At 10 Hz, groups [0..4] and [5..9] are the exact two Conv1d stem supports.
    assert np.allclose(labels, [2.0 / 9.0, 7.0 / 9.0])


def test_resume_restores_optimizer_and_head_state(tmp_path) -> None:
    runner = _module()
    config = ActionFormerConfig(
        d_model=8,
        num_heads=2,
        attention_window=8,
        pyramid_levels=2,
        regression_ranges_seconds=((0.0, 45.0), (30.0, float("inf"))),
    )
    model = ActionFormerHighlightModel(config)
    head = torch.nn.Conv1d(8, 1, 1)
    optimizer = AdamW(list(model.backbone.parameters()) + list(head.parameters()), lr=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=3)
    checkpoint = tmp_path / "last.pt"
    metadata = {
        "feature_schema_version": "1.1",
        "channel_order": ["rms", "pitch", "silence", "text_score", "scene_change", "gesture", "turn_rate"],
        "dataset_fingerprint": "x",
        "split_fingerprint": "y",
        "normalization_policy_version": "test",
    }
    save_actionformer_checkpoint(
        checkpoint, model, metadata=metadata, training_state=runner._state(2, optimizer, scheduler, head, 0.3, 1)
    )
    restored_model = ActionFormerHighlightModel(config)
    restored_head = torch.nn.Conv1d(8, 1, 1)
    restored_optimizer = AdamW(list(restored_model.backbone.parameters()) + list(restored_head.parameters()), lr=1e-3)
    restored_scheduler = CosineAnnealingLR(restored_optimizer, T_max=3)
    start, best, stale = runner._resume(
        checkpoint, restored_model, restored_head, restored_optimizer, restored_scheduler, torch.device("cpu")
    )
    assert (start, best, stale) == (3, 0.3, 1)
    assert all(torch.equal(a, b) for a, b in zip(head.parameters(), restored_head.parameters()))
