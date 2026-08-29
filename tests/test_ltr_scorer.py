from __future__ import annotations

import pytest
import torch

from highlight_agent.models.ltr_scorer import AdditiveAttentionScorer


def test_forward_shape():
    model = AdditiveAttentionScorer()
    x = torch.randn(8, 7)
    out = model(x)
    assert out.shape == (8, 1)


def test_forward_dtype():
    model = AdditiveAttentionScorer()
    x = torch.randn(4, 7)
    assert model(x).dtype == torch.float32


def test_save_load_identical_scores(tmp_path):
    model = AdditiveAttentionScorer()
    model.eval()
    x = torch.randn(4, 7)
    scores_before = model(x).detach()
    p = tmp_path / "model.pt"
    model.save(p, metadata={"val_ap": 0.5, "L_ref": 40.0})
    loaded = AdditiveAttentionScorer.load(p)
    scores_after = loaded(x).detach()
    assert torch.allclose(scores_before, scores_after, atol=1e-5)


def test_load_checkpoint_returns_metadata_and_requested_device(tmp_path):
    model = AdditiveAttentionScorer()
    path = tmp_path / "nested" / "model.pt"
    model.save(path, metadata={"L_ref": 36.0})

    loaded, metadata = AdditiveAttentionScorer.load_checkpoint(path, device="cpu")

    assert next(loaded.parameters()).device.type == "cpu"
    assert metadata["L_ref"] == 36.0
    assert loaded.training is False


def test_load_wrong_dim_raises(tmp_path):
    model = AdditiveAttentionScorer(in_features=7)
    p = tmp_path / "model.pt"
    model.save(p)
    import torch as _t
    ckpt = _t.load(p, map_location="cpu", weights_only=False)
    ckpt["in_features"] = 14
    _t.save(ckpt, p)
    with pytest.raises(ValueError, match="expects 14 features"):
        AdditiveAttentionScorer.load(p)


def test_checkpoint_has_metadata(tmp_path):
    model = AdditiveAttentionScorer()
    p = tmp_path / "m.pt"
    model.save(p, metadata={"epoch": 5})
    ckpt = torch.load(p, map_location="cpu", weights_only=False)
    assert "state_dict" in ckpt
    assert "in_features" in ckpt
    assert "hidden_dim" in ckpt
    assert "metadata" in ckpt


def test_xavier_init():
    model = AdditiveAttentionScorer()
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            assert not torch.all(m.weight == 0)
