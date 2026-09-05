import pytest
import torch

from highlight_agent.models.actionformer import (
    ActionFormerConfig,
    ActionFormerHighlightModel,
    TemporalProposal,
    actionformer_losses,
    decode_proposals,
    interval_diou_loss,
    load_actionformer_checkpoint,
    save_actionformer_checkpoint,
    soft_nms,
)


def _config(**overrides) -> ActionFormerConfig:
    values = {
        "d_model": 16,
        "num_heads": 4,
        "attention_window": 16,
        "pyramid_levels": 2,
        "blocks_per_level": 1,
        "head_depth": 1,
        "dropout": 0.0,
        "regression_ranges_seconds": ((0.0, 45.0), (30.0, float("inf"))),
    }
    values.update(overrides)
    return ActionFormerConfig(**values)


def test_actionformer_shapes_and_padding_mask() -> None:
    config = _config()
    model = ActionFormerHighlightModel(config)
    features = torch.rand(2, 7, 200)
    valid_mask = torch.ones(2, 200, dtype=torch.bool)
    valid_mask[1, 150:] = False

    output = model(features, valid_mask)

    assert [item.shape for item in output["logits"]] == [(2, 40, 1), (2, 20, 1)]
    assert [item.shape for item in output["offsets"]] == [(2, 40, 2), (2, 20, 2)]
    assert output["masks"][0][1].sum().item() == 30
    assert torch.isfinite(output["features"][0]).all()


def test_actionformer_loss_is_finite_and_backpropagates() -> None:
    config = _config()
    model = ActionFormerHighlightModel(config)
    features = torch.rand(1, 7, 600)
    output = model(features)
    losses = actionformer_losses(
        output,
        [torch.tensor([[20.0, 50.0]])],
        config,
    )

    assert losses["positive_points"].item() > 0
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_interval_diou_is_zero_for_identical_intervals() -> None:
    locations = torch.tensor([10.0])
    offsets = torch.tensor([[3.0, 5.0]])
    assert interval_diou_loss(locations, offsets, offsets).item() == 0.0


def test_decoder_enforces_duration_and_video_bounds() -> None:
    config = _config(score_threshold=0.5, pre_nms_topk=10)
    outputs = {
        "logits": [torch.tensor([[[10.0], [-10.0]]]), torch.tensor([[[-10.0]]])],
        "offsets": [torch.tensor([[[80.0, 80.0], [1.0, 1.0]]]), torch.ones(1, 1, 2)],
        "masks": [torch.ones(1, 2, dtype=torch.bool), torch.ones(1, 1, dtype=torch.bool)],
        "features": [torch.zeros(1, 16, 2), torch.zeros(1, 16, 1)],
    }

    proposals = decode_proposals(outputs, config, video_durations=[100.0])

    assert len(proposals) == 1
    assert proposals[0].start == 0.0
    assert proposals[0].end <= 100.0
    assert 30 <= proposals[0].duration <= 90


def test_soft_nms_keeps_diverse_proposals_and_decays_overlap() -> None:
    proposals = [
        TemporalProposal(0, 40, 0.9, 0, 1),
        TemporalProposal(2, 42, 0.8, 0, 2),
        TemporalProposal(60, 100, 0.7, 0, 3),
    ]

    selected = soft_nms(proposals, sigma=0.5, top_k=3)

    assert selected[0].start == 0
    assert selected[1].start == 60
    assert selected[2].score < 0.8


def test_actionformer_checkpoint_round_trip(tmp_path) -> None:
    config = _config()
    model = ActionFormerHighlightModel(config)
    path = tmp_path / "model.pt"
    metadata = {
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
    }

    save_actionformer_checkpoint(path, model, metadata=metadata)
    loaded, loaded_metadata, proposal_state = load_actionformer_checkpoint(path)

    assert loaded.config == config
    assert loaded_metadata == metadata
    assert proposal_state is None


def test_actionformer_checkpoint_rejects_incompatible_feature_contract(tmp_path) -> None:
    model = ActionFormerHighlightModel(_config())
    path = tmp_path / "model.pt"
    metadata = {
        "feature_schema_version": "0.9",
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
    }
    save_actionformer_checkpoint(path, model, metadata=metadata)

    with pytest.raises(ValueError, match="feature schema"):
        load_actionformer_checkpoint(path)
