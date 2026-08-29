from __future__ import annotations

import json
from pathlib import Path

from scripts.run_in_domain_pilot import PROJECT_ROOT, _load_catalog, write_fold_manifests


def test_catalog_contains_only_completed_annotation_videos() -> None:
    catalog = _load_catalog(PROJECT_ROOT / "data/manifests/in_domain_pilot.json")

    assert len(catalog["videos"]) == 6
    assert {video["domain"] for video in catalog["videos"]} == {"lecture", "podcast"}
    assert all(Path(video["annotation_path"]).suffix == ".csv" for video in catalog["videos"])


def test_fold_manifests_are_disjoint_and_complete(tmp_path) -> None:
    paths = write_fold_manifests(PROJECT_ROOT / "data/manifests/in_domain_pilot.json", tmp_path)

    assert len(paths) == 6
    for path in paths:
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        splits = {record["split"] for record in records}
        assert len(records) == 6
        assert splits == {"train", "val", "test"}
        assert sum(record["split"] == "train" for record in records) == 4
        assert sum(record["split"] == "val" for record in records) == 1
        assert sum(record["split"] == "test" for record in records) == 1
        assert all(record["source"] == "in_domain_ordinal" for record in records)
        assert all(not Path(record["video_path"]).is_absolute() for record in records)
