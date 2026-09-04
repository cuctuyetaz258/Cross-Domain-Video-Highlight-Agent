from pathlib import Path

import pytest
import torch

from highlight_agent.features.ltr_contract import (
    LTR_FEATURE_SCHEMA_VERSION,
    LTRPipelineError,
    feature_contract,
)
from highlight_agent.models.tcn_ltr_scorer import TemporalConvLTRScorer


def _metadata() -> dict:
    return {
        "schema_version": LTR_FEATURE_SCHEMA_VERSION,
        "feature_schema": feature_contract(),
        "L_ref": 60.0,
        "dataset_fingerprint": "fixture",
        "epoch": 1,
    }


def test_tcn_scores_sequences_and_preserves_length() -> None:
    model = TemporalConvLTRScorer()
    assert model.receptive_field_tokens == 31
    assert model(torch.zeros(9, 7)).shape == (9,)
    assert model(torch.zeros(2, 9, 7)).shape == (2, 9)


def test_tcn_checkpoint_round_trip_and_preflight(tmp_path: Path) -> None:
    path = tmp_path / "tcn.pt"
    model = TemporalConvLTRScorer()
    model.save(path, _metadata())
    loaded, metadata = TemporalConvLTRScorer.load_checkpoint(path)
    assert loaded.hidden_dim == 32
    assert metadata["L_ref"] == 60.0
    assert TemporalConvLTRScorer.preflight(path)["model_type"] == "tcn_ltr_v2"


def test_tcn_preflight_rejects_an_mlp_checkpoint(tmp_path: Path) -> None:
    path = tmp_path / "not-a-tcn.pt"
    torch.save({"checkpoint_version": "1.1"}, path)
    with pytest.raises(LTRPipelineError, match="LOAD_FAILED"):
        TemporalConvLTRScorer.preflight(path)
