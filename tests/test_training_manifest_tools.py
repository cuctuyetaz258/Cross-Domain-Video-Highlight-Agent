from __future__ import annotations

import json

import pytest

from scripts.build_fusion_dataset import build_fusion_dataset, candidate_importance
from scripts.compose_training_manifests import compose_manifests


def test_compose_manifests_preserves_sources_and_splits(tmp_path):
    tvsum = tmp_path / "tvsum.jsonl"
    summe = tmp_path / "summe.jsonl"
    tvsum.write_text(
        json.dumps(
            {
                "video_id": "tv",
                "source": "tvsum",
                "domain": "benchmark",
                "split": "train",
                "video_path": "tv.mp4",
            }
        ),
        encoding="utf-8",
    )
    summe.write_text(
        json.dumps(
            {
                "video_id": "sm",
                "source": "summe",
                "domain": "benchmark",
                "split": "val",
                "video_path": "sm.mp4",
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "combined.jsonl"

    report = compose_manifests([tvsum, summe], output)

    assert report["sources"] == {"summe": 1, "tvsum": 1}
    assert report["splits"] == {"train": 1, "val": 1}
    assert len(output.read_text(encoding="utf-8").splitlines()) == 2


def test_compose_manifests_rejects_duplicate_video(tmp_path):
    paths = []
    for index in range(2):
        path = tmp_path / f"manifest-{index}.jsonl"
        path.write_text(
            json.dumps(
                {
                    "video_id": "same",
                    "source": f"source-{index}",
                    "domain": "benchmark",
                    "split": "train",
                }
            ),
            encoding="utf-8",
        )
        paths.append(path)

    with pytest.raises(ValueError, match="appears"):
        compose_manifests(paths, tmp_path / "output.jsonl")


def test_candidate_importance_is_overlap_weighted():
    score = candidate_importance(
        1.0,
        5.0,
        [[0.0, 2.0, 1.0], [2.0, 4.0, 3.0], [4.0, 6.0, 5.0]],
    )

    assert score == pytest.approx(3.0)


def test_build_fusion_dataset_joins_metadata_and_custom_scores(tmp_path):
    manifest = tmp_path / "custom.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "video_id": "lecture-1",
                "source": "custom_scores",
                "domain": "lecture",
                "split": "val",
                "importance_segments": [[0.0, 2.0, 1.0], [2.0, 4.0, 5.0]],
            }
        ),
        encoding="utf-8",
    )
    metadata = tmp_path / "metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "video_id": "lecture-1",
                "pipeline": {
                    "checkpoint": {"fingerprint": "checkpoint"},
                    "llm_run": {
                        "applied": True,
                        "model": "gpt-4o-mini",
                        "prompt_version": "v2",
                    },
                    "fusion_candidates": [
                        {
                            "candidate_id": "c1",
                            "start_time": 0.0,
                            "end_time": 4.0,
                            "ltr_score": 0.2,
                            "llm_score": 0.8,
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "fusion.jsonl"

    report = build_fusion_dataset(
        manifest_path=manifest,
        metadata_paths=[metadata],
        output_path=output,
    )

    row = json.loads(output.read_text(encoding="utf-8"))
    assert report["candidate_count"] == 1
    assert row["target_importance"] == pytest.approx(3.0)
    assert row["ltr_checkpoint_fingerprint"] == "checkpoint"
