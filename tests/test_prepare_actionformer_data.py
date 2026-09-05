from scripts.prepare_actionformer_data import assign_stratified_folds, normalize_boundary


def test_normalize_boundary_enforces_duration_range() -> None:
    importance = [(float(index), float(index + 2), 5.0 if 40 <= index < 70 else 1.0) for index in range(0, 120, 2)]

    short = normalize_boundary(10, 25, video_duration=120, importance=importance)
    long = normalize_boundary(0, 110, video_duration=120, importance=importance)

    assert short[1] - short[0] == 30
    assert long[1] - long[0] == 90
    assert long[2] == "cropped_to_best_max_window"


def test_stratified_five_fold_covers_each_video_once() -> None:
    records = [
        {"video_id": f"lecture-{index}", "domain": "lecture"}
        for index in range(10)
    ] + [
        {"video_id": f"podcast-{index}", "domain": "podcast"}
        for index in range(8)
    ]

    manifests = assign_stratified_folds(records, folds=5, seed=42)
    test_ids = [
        item["video_id"]
        for manifest in manifests.values()
        for item in manifest
        if item["split"] == "test"
    ]

    assert sorted(test_ids) == sorted(item["video_id"] for item in records)
    assert sorted(sum(item["split"] == "test" for item in manifest) for manifest in manifests.values()) == [3, 3, 4, 4, 4]
