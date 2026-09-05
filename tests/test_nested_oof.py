from dataclasses import replace

import numpy as np
import pytest
import torch

from highlight_agent.models.actionformer import ActionFormerConfig, ActionFormerHighlightModel, TemporalProposal
from highlight_agent.models.oof_proposals import load_nested_proposal_cache, load_oof_proposal_cache
from highlight_agent.models.proposal_ltr import ProposalLTRConfig
from highlight_agent.models.proposal_protocol import assert_lineage_allowed, example_digest
from highlight_agent.models.train_actionformer_ltr import ActionFormerExample, _evaluate_proposal_ltr
from scripts.run_actionformer_nested_cv import inner_splits, run_outer_fold


def example(video_id):
    return ActionFormerExample(
        video_id, "lecture", 60.0, np.random.default_rng(5).random((7, 600), dtype=np.float32),
        np.asarray([[10, 40]], dtype=np.float32), np.asarray([[0, 30, 1], [30, 60, 5]], dtype=np.float32),
    )


def test_nested_split_never_selects_checkpoint_on_prediction_target():
    rows = [example(str(i)) for i in range(10)]
    seen = []
    for fit, selection, targets in inner_splits(rows):
        fit_ids, val_ids, target_ids = ({x.video_id for x in group} for group in (fit, selection, targets))
        assert fit_ids.isdisjoint(val_ids | target_ids)
        assert val_ids.isdisjoint(target_ids)
        assert fit_ids | val_ids | target_ids == {str(i) for i in range(10)}
        seen.extend(target_ids)
    assert sorted(seen) == [str(i) for i in range(10)]


@pytest.mark.parametrize("field", ["train_video_ids", "selection_video_ids"])
def test_recursive_lineage_catches_indirect_test_leakage(field):
    parent = {"train_video_ids": [], "selection_video_ids": [], "ancestors": []}
    parent[field] = ["outer-test"]
    lineage = {"train_video_ids": ["train"], "selection_video_ids": [], "ancestors": [parent]}
    with pytest.raises(ValueError, match="outer-test"):
        assert_lineage_allowed(lineage, {"train"})


def test_content_hash_changes_without_shape_change():
    row = example("v")
    changed = row.features.copy()
    changed[0, 0] += 0.1
    assert example_digest(row) != example_digest(replace(row, features=changed))


def test_shared_oof_rejected_by_default(tmp_path):
    with pytest.raises(ValueError, match="unsafe"):
        load_oof_proposal_cache(tmp_path / "unused.json")


@pytest.mark.parametrize("empty", [True, False])
def test_validation_never_adds_ground_truth_candidates(monkeypatch, empty):
    import highlight_agent.models.train_actionformer_ltr as training

    proposals = [] if empty else [TemporalProposal(0, 30, .8, 0, 0), TemporalProposal(30, 60, .7, 0, 1)]
    monkeypatch.setattr(training, "decode_proposals", lambda *args, **kwargs: list(proposals))

    def forbidden(*args, **kwargs):
        raise AssertionError("GT/grid candidates leaked into validation")

    monkeypatch.setattr(training, "proposal_training_set", forbidden)

    class Scorer(torch.nn.Module):
        def forward(self, features, lists, **kwargs):
            assert lists == [proposals]
            return features.new_tensor([0., 1.]), [(0, 0), (0, 1)]

    model = ActionFormerHighlightModel(ActionFormerConfig(d_model=8, attention_window=16))
    kwargs = dict(device=torch.device("cpu"), margin=1., utility_delta=.1, loss_type="ranknet",
                  pair_weighting="utility", ndcg_k=3, gain_scale=4., max_pairs_per_video=256)
    result = _evaluate_proposal_ltr(model, Scorer(), [example("val")], **kwargs)
    repeated = _evaluate_proposal_ltr(model, Scorer(), [example("val")], **kwargs)
    assert result == repeated
    assert result["candidate_source"] == "predicted_only"
    assert result["empty_candidate_videos"] == int(empty)
    assert result["ndcg_at_3"] == (0. if empty else 1.)


def test_nested_run_roundtrip_and_cache_mismatch_checks(tmp_path):
    train = [example(str(i)) for i in range(6)]
    val, test = [example("val")], [example("test")]
    directory = tmp_path / "run"
    result = run_outer_fold(
        train=train, val=val, test=test, directory=directory, fold=0,
        config=ActionFormerConfig(d_model=8, attention_window=16, score_threshold=0., pre_nms_topk=5),
        scorer_config=ProposalLTRConfig(d_model=8, num_heads=2, ffn_dim=16, num_inducing_points=2),
        loc_epochs=1, ltr_epochs=1, patience=1,
    )
    assert result["status"] == "complete"
    for name in ("inner0", "inner1", "inner2", "outer_localization", "imsab"):
        for filename in ("best.pt", "last.pt", "train_log.json", "curves.svg", "history.csv"):
            assert (directory / name / filename).is_file()
    kwargs = dict(train_examples=train, val_video_ids=["val"], test_video_ids=["test"])
    cache_path = directory / "nested_proposals.json"
    proposals, metadata = load_nested_proposal_cache(cache_path, **kwargs)
    assert set(proposals) == {str(i) for i in range(6)}
    assert metadata["outer_fold"] == 0
    with pytest.raises(ValueError, match="different outer split"):
        load_nested_proposal_cache(cache_path, **{**kwargs, "test_video_ids": ["other"]})
    changed = train[0].importance.copy()
    changed[0, 2] += .1
    with pytest.raises(ValueError, match="content fingerprint"):
        load_nested_proposal_cache(cache_path, **{**kwargs, "train_examples": [replace(train[0], importance=changed), *train[1:]]})
    checkpoint = directory / "inner0" / "best.pt"
    with checkpoint.open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="checkpoint fingerprint"):
        load_nested_proposal_cache(cache_path, **kwargs)
