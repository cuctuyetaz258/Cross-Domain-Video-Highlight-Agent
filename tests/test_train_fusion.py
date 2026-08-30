from __future__ import annotations

import pytest

from evaluation.train_fusion import fit_global_alpha, macro_domain_ndcg
from highlight_agent.llm.fusion import FusionCalibrator, fuse_ranked_scores, percentile_rank


def _records():
    records = []
    for domain in ("lecture", "podcast"):
        for index, relevance in enumerate((5.0, 3.0, 1.0)):
            records.append(
                {
                    "video_id": f"{domain}-video",
                    "domain": domain,
                    "candidate_id": f"{domain}-{index}",
                    "ltr_score": relevance,
                    "llm_score": 6.0 - relevance,
                    "target_importance": relevance,
                    "split": "val",
                }
            )
    return records


def test_percentile_rank_preserves_ties_and_range():
    assert percentile_rank([9, 5, 1]).tolist() == [1.0, 0.5, 0.0]
    assert percentile_rank([2, 2, 1]).tolist() == [0.75, 0.75, 0.0]
    assert percentile_rank([4]).tolist() == [0.5]


def test_rank_fusion_uses_one_alpha():
    result = fuse_ranked_scores([9, 5, 1], [1, 5, 9], alpha=0.75)
    assert result.tolist() == pytest.approx([0.75, 0.5, 0.25])


def test_grid_search_selects_ltr_when_it_matches_both_domains():
    result = fit_global_alpha(_records(), step=0.25, k=3)

    assert result["best"]["alpha"] == 1.0
    assert result["best"]["macro_ndcg"] == pytest.approx(1.0)
    macro, domains, videos = macro_domain_ndcg(_records(), alpha=1.0, k=3)
    assert macro == pytest.approx(1.0)
    assert domains == {"lecture": pytest.approx(1.0), "podcast": pytest.approx(1.0)}
    assert set(videos) == {"lecture-video", "podcast-video"}


def test_calibrator_rejects_wrong_checkpoint(tmp_path):
    path = tmp_path / "fusion.json"
    payload = FusionCalibrator(
        alpha=0.7,
        selection_metric="macro_ndcg@3",
        ltr_checkpoint_fingerprint="expected",
        llm_model="gpt-4o-mini",
        prompt_version="v2",
    ).to_dict()
    import json

    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="checkpoint"):
        FusionCalibrator.load(path, expected_checkpoint_fingerprint="other")
