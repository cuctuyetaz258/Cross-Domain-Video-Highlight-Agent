from __future__ import annotations

import csv

import pytest

from scripts.prepare_custom_manifest import (
    _align_annotation_to_media,
    assign_group_folds,
    load_completed_annotation,
)


def _write_annotation(path, *, video_id: str, domain: str, scores: list[int]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["video_id", "start_sec", "end_sec", "importance", "domain"],
        )
        writer.writeheader()
        for index, score in enumerate(scores):
            writer.writerow(
                {
                    "video_id": video_id,
                    "start_sec": index * 2,
                    "end_sec": (index + 1) * 2,
                    "importance": score,
                    "domain": domain,
                }
            )


def test_load_completed_annotation_keeps_importance_timeline(tmp_path):
    path = tmp_path / "lecture.csv"
    _write_annotation(path, video_id="lecture-1", domain="lecture", scores=[1, 3, 5])

    record = load_completed_annotation(path)

    assert record["source"] == "custom_scores"
    assert record["duration"] == 6.0
    assert record["importance_segments"] == [
        [0.0, 2.0, 1.0],
        [2.0, 4.0, 3.0],
        [4.0, 6.0, 5.0],
    ]


def test_load_completed_annotation_rejects_unfinished_csv(tmp_path):
    path = tmp_path / "lecture.csv"
    _write_annotation(path, video_id="lecture-1", domain="lecture", scores=[1, 2])
    text = path.read_text(encoding="utf-8-sig").replace(",2,lecture", ",,lecture")
    path.write_text(text, encoding="utf-8-sig")

    with pytest.raises(ValueError, match="no importance"):
        load_completed_annotation(path)


def test_align_annotation_uses_media_duration_and_clips_tail():
    record = {
        "duration": 6.0,
        "importance_segments": [
            [0.0, 2.0, 1.0],
            [2.0, 4.0, 3.0],
            [4.0, 6.0, 5.0],
        ],
    }

    aligned = _align_annotation_to_media(record, 5.25)

    assert aligned["duration"] == 5.25
    assert aligned["annotation_duration"] == 6.0
    assert aligned["importance_segments"][-1] == [4.0, 5.25, 5.0]
    assert aligned["unlabeled_tail_seconds"] == 0.0


def test_align_annotation_marks_unlabeled_media_tail_as_ignored():
    record = {
        "duration": 4.0,
        "importance_segments": [[0.0, 2.0, 1.0], [2.0, 4.0, 5.0]],
    }

    aligned = _align_annotation_to_media(record, 5.25)

    assert aligned["importance_segments"][-1] == [4.0, 5.25, 3.0]
    assert aligned["unlabeled_tail_policy"] == "ignored_score_3"
    assert aligned["unlabeled_tail_seconds"] == pytest.approx(1.25)


def test_assign_group_folds_keeps_each_video_in_one_split():
    records = [
        {"video_id": f"{domain}-{index}", "domain": domain}
        for domain in ("lecture", "podcast")
        for index in range(5)
    ]

    assigned = assign_group_folds(records, fold=0, folds=5, seed=42)

    for domain in ("lecture", "podcast"):
        domain_records = [record for record in assigned if record["domain"] == domain]
        assert sum(record["split"] == "train" for record in domain_records) == 3
        assert sum(record["split"] == "val" for record in domain_records) == 1
        assert sum(record["split"] == "test" for record in domain_records) == 1
    assert len({record["video_id"] for record in assigned}) == len(assigned)
