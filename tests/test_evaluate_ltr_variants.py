from __future__ import annotations

import json

import numpy as np
import pytest

from evaluation.evaluate_ltr_variants import (
    FULL_LTR_VARIANT,
    LLM_FAILURE_VARIANT,
    LLM_SUCCESS_VARIANT,
    classify_llm_run,
    evaluate_ltr_model_variants,
    summarize_llm_runs,
)
from highlight_agent.features.ltr_contract import feature_contract
from highlight_agent.models.ltr_scorer import AdditiveAttentionScorer
from highlight_agent.models.train_offline import feature_cache_metadata


def _write_model_fixture(tmp_path):
    cache_root = tmp_path / "cache"
    video_cache = cache_root / "video-01"
    video_cache.mkdir(parents=True)
    matrix = np.random.default_rng(42).random((7, 200), dtype=np.float32)
    np.save(video_cache / "feature_matrix.npy", matrix)
    (video_cache / "metadata.json").write_text(
        json.dumps(feature_cache_metadata("video-01", matrix)),
        encoding="utf-8",
    )

    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "video_id": "video-01",
                "domain": "lecture",
                "source": "custom",
                "split": "val",
                "duration": 20.0,
                "relevant_windows": [[10.0, 15.0]],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    checkpoint = tmp_path / "ltr.pt"
    AdditiveAttentionScorer().save(
        checkpoint,
        metadata={
            "schema_version": "1.1",
            "feature_schema": feature_contract(),
            "L_ref": 30.0,
            "dataset_fingerprint": "fixture-dataset",
            "selection_ap": 0.75,
            "epoch": 2,
        },
    )
    return manifest, cache_root, checkpoint


def test_model_report_separates_full_ltr_and_channel_sensitivity(tmp_path) -> None:
    manifest, cache_root, checkpoint = _write_model_fixture(tmp_path)

    report = evaluate_ltr_model_variants(
        manifest=manifest,
        cache_dir=cache_root,
        checkpoint=checkpoint,
        split="val",
        device="cpu",
        channels=["rms", "gesture"],
        top_k=3,
    )

    variants = report["variants"]
    assert [variant["variant_key"] for variant in variants] == [
        FULL_LTR_VARIANT,
        "ltr_without_rms",
        "ltr_without_gesture",
    ]
    assert variants[0]["variant_group"] == "full_ltr"
    assert variants[0]["diagnostic_only"] is False
    assert all(variant["status"] == "completed" for variant in variants)
    assert all(
        variant["checkpoint_fingerprint"] == report["checkpoint_fingerprint"]
        for variant in variants
    )
    assert variants[1]["ablation"] == {
        "type": "zeroed_channel",
        "removed_channels": ["rms"],
        "interpretation": "channel_sensitivity_under_distribution_shift",
    }
    assert not any(
        "random" in variant["variant_key"] or "profile" in variant["variant_key"]
        for variant in variants
    )


def test_model_report_rejects_unknown_or_duplicate_channels(tmp_path) -> None:
    manifest, cache_root, checkpoint = _write_model_fixture(tmp_path)
    common = {
        "manifest": manifest,
        "cache_dir": cache_root,
        "checkpoint": checkpoint,
        "split": "val",
        "device": "cpu",
    }

    with pytest.raises(ValueError, match="unknown feature channels"):
        evaluate_ltr_model_variants(**common, channels=["unknown"])
    with pytest.raises(ValueError, match="must not contain duplicates"):
        evaluate_ltr_model_variants(**common, channels=["rms", "rms"])


def _write_run_metadata(tmp_path, name: str, payload: dict):
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_llm_success_and_failure_are_classified_separately(tmp_path) -> None:
    success_path = _write_run_metadata(
        tmp_path,
        "success",
        {
            "pipeline": {
                "mode": "ltr_llm_rerank",
                "checkpoint": {"fingerprint": "abc"},
                "llm_run": {
                    "enabled": True,
                    "applied": True,
                    "provider": "openai",
                    "model": "fake",
                    "assessed_count": 3,
                },
            },
            "highlights": [
                {"title": "Complete idea", "summary": "Summary", "completeness_score": 0.9}
            ],
        },
    )
    failure_path = _write_run_metadata(
        tmp_path,
        "failure",
        {
            "pipeline": {
                "mode": "ltr_required",
                "checkpoint": {"fingerprint": "abc"},
                "llm_run": {
                    "enabled": True,
                    "applied": False,
                    "provider": "openai",
                    "fallback_reason": "provider timeout",
                },
            },
            "highlights": [{"title": None, "summary": None}],
        },
    )

    success = classify_llm_run(success_path)
    failure = classify_llm_run(failure_path)
    summary = summarize_llm_runs([success, failure])

    assert success["variant_key"] == LLM_SUCCESS_VARIANT
    assert success["ranking_source"] == "ltr_plus_llm"
    assert success["status"] == "completed"
    assert failure["variant_key"] == LLM_FAILURE_VARIANT
    assert failure["ranking_source"] == "ltr"
    assert failure["fallback_reason"] == "provider timeout"
    assert summary[LLM_SUCCESS_VARIANT]["run_count"] == 1
    assert summary[LLM_FAILURE_VARIANT]["run_count"] == 1
    assert summary[LLM_FAILURE_VARIANT]["fallback_reasons"] == {"provider timeout": 1}


def test_llm_failure_without_reason_is_reported_as_failed_metadata(tmp_path) -> None:
    path = _write_run_metadata(
        tmp_path,
        "invalid_failure",
        {
            "pipeline": {
                "mode": "ltr_required",
                "llm_run": {"enabled": True, "applied": False},
            }
        },
    )

    run = classify_llm_run(path)

    assert run["variant_key"] == LLM_FAILURE_VARIANT
    assert run["status"] == "failed"
    assert "fallback_reason" in run["failure_reason"]


def test_missing_llm_runs_stay_not_run() -> None:
    summary = summarize_llm_runs([])

    assert summary[LLM_SUCCESS_VARIANT]["status"] == "not_run"
    assert summary[LLM_FAILURE_VARIANT]["status"] == "not_run"
    assert summary[LLM_SUCCESS_VARIANT]["run_count"] == 0
