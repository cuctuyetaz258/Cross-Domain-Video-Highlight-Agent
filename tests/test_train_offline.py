from __future__ import annotations

import json
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import torch

from highlight_agent.models.train_offline import (
    FEATURE_CHANNELS,
    WindowExample,
    build_balanced_epoch_pairs,
    create_pairwise_dataset,
    create_window_labels,
    evaluate_average_precision,
    feature_cache_metadata,
    load_feature_matrix,
    load_qvhighlights,
    load_training_manifest,
    load_tvsum,
    margin_ranking_loss,
    temporal_smoothness_loss,
    train,
)


def _write_cache(cache_root, video_id: str, matrix: np.ndarray, *, metadata: bool = True) -> None:
    cache_dir = cache_root / video_id
    cache_dir.mkdir(parents=True)
    np.save(cache_dir / "feature_matrix.npy", matrix)
    if metadata:
        (cache_dir / "metadata.json").write_text(
            json.dumps(feature_cache_metadata(video_id, matrix)),
            encoding="utf-8",
        )


def _qv_record(video_id: str, duration: float = 20.0) -> dict:
    return {
        "video_id": video_id,
        "domain": "lecture",
        "source": "qvhighlights",
        "duration": duration,
        "relevant_windows": [[10.0, 15.0]],
        "saliency_scores": [],
    }


def _example(video_id: str, index: int, value: float, label: int) -> WindowExample:
    feature = np.zeros(len(FEATURE_CHANNELS), dtype=np.float32)
    feature[0] = value
    return WindowExample(
        video_id=video_id,
        domain="lecture",
        window_index=index,
        start=float(index),
        end=float(index + 1),
        feature=feature,
        label=label,
        score=value,
    )


def test_iou_perfect():
    record = {
        "source": "qvhighlights",
        "duration": 5.0,
        "relevant_windows": [[0.0, 5.0]],
        "saliency_scores": [[0, 0, 3]],
    }

    labels = create_window_labels(record, window_sec=5.0, hop_sec=1.0)

    assert labels[0]["label"] == "positive"
    assert labels[0]["score"] == 1.0


def test_iou_no_overlap():
    record = {
        "source": "qvhighlights",
        "duration": 15.0,
        "relevant_windows": [[10.0, 15.0]],
        "saliency_scores": [[0, 0, 3]],
    }

    labels = create_window_labels(record, window_sec=5.0, hop_sec=1.0)

    assert labels[0]["label"] == "negative"
    assert labels[0]["score"] == 0.0


def test_load_qvhighlights_prefers_explicit_duration(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text(
        json.dumps({"vid": "vid1", "duration": 30.0, "relevant_windows": [[0, 5]]}) + "\n",
        encoding="utf-8",
    )

    records = load_qvhighlights(path)

    assert records[0]["video_id"] == "vid1"
    assert records[0]["duration"] == 30.0


def test_load_tvsum_preserves_dataset_category(monkeypatch) -> None:
    import scipy.io

    row = {
        "video": np.asarray(["vehicle-video"], dtype=object),
        "category": np.asarray(["VT"], dtype=object),
        "annotations": np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
    }
    monkeypatch.setattr(
        scipy.io,
        "loadmat",
        lambda _: {"tvsum50": np.asarray([[row]], dtype=object)},
    )

    records = load_tvsum("tvsum50.mat")

    assert records[0]["dataset"] == "tvsum"
    assert records[0]["category"] == "VT"
    assert records[0]["domain"] == "benchmark"


def test_load_training_manifest_filters_split(tmp_path):
    path = tmp_path / "manifest.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"video_id": "train-video", "split": "train"}),
                json.dumps({"video_id": "val-video", "split": "val"}),
            ]
        ),
        encoding="utf-8",
    )

    records = load_training_manifest(path, split="val")

    assert [record["video_id"] for record in records] == ["val-video"]


def test_custom_long_highlight_uses_window_coverage_not_symmetric_iou():
    record = {
        "source": "custom_pseudo",
        "duration": 90.0,
        "relevant_windows": [[20.0, 80.0]],
    }

    labels = create_window_labels(record, window_sec=5.0, hop_sec=5.0)

    assert labels[0]["label"] == "negative"
    assert next(item for item in labels if item["start"] == 20.0)["label"] == "positive"


def test_custom_scores_preserve_graded_window_supervision():
    record = {
        "source": "custom_scores",
        "duration": 10.0,
        "importance_segments": [
            [0.0, 2.0, 1.0],
            [2.0, 4.0, 2.0],
            [4.0, 6.0, 3.0],
            [6.0, 8.0, 4.0],
            [8.0, 10.0, 5.0],
        ],
    }

    labels = create_window_labels(record, window_sec=2.0, hop_sec=2.0)

    assert [label["label"] for label in labels] == [
        "negative",
        "negative",
        "ignored",
        "positive",
        "positive",
    ]
    assert [label["score"] for label in labels] == [1, 2, 3, 4, 5]


def test_balanced_epoch_pairs_caps_video_and_applies_source_weights():
    examples = []
    for source, video_id in (("tvsum", "tv"), ("summe", "sm")):
        for index in range(4):
            example = _example(video_id, index, float(index), int(index >= 2))
            examples.append(
                WindowExample(**{**example.__dict__, "source": source})
            )

    pairs, report = build_balanced_epoch_pairs(
        examples,
        source_weights={"tvsum": 0.75, "summe": 0.25},
        max_pairs_per_video=4,
        seed=7,
    )

    assert len(pairs) == 8
    assert report["video_pair_counts"] == {"sm": 4, "tv": 4}
    assert report["source_pair_counts"] == {"summe": 2, "tvsum": 6}
    assert all(positive.video_id == negative.video_id for positive, negative in pairs)


def test_create_window_labels_summe_has_both_classes():
    record = {
        "source": "summe",
        "fps": 25.0,
        "frame_scores": np.linspace(0, 1, 500, dtype=np.float32),
    }

    labels = create_window_labels(record, window_sec=5.0, hop_sec=1.0)

    assert any(label["label"] == "positive" for label in labels)
    assert any(label["label"] == "negative" for label in labels)


def test_benchmark_labels_do_not_extend_past_media_duration():
    record = {
        "source": "summe",
        "fps": 30.0,
        "duration": 103.966667,
        "frame_scores": np.linspace(0, 1, 3120, dtype=np.float32),
    }

    labels = create_window_labels(record, window_sec=5.0, hop_sec=1.0)

    assert max(label["end"] for label in labels) <= 103.9


def test_feature_cache_requires_canonical_shape_dtype_and_metadata(tmp_path):
    valid = np.zeros((7, 100), dtype=np.float32)
    _write_cache(tmp_path, "valid", valid)
    assert load_feature_matrix(tmp_path, "valid").shape == (7, 100)

    transposed = np.zeros((100, 7), dtype=np.float32)
    _write_cache(tmp_path, "transposed", transposed)
    with pytest.raises(ValueError, match=r"shape \(7, T\)"):
        load_feature_matrix(tmp_path, "transposed")

    _write_cache(tmp_path, "float64", np.zeros((7, 100), dtype=np.float64))
    with pytest.raises(ValueError, match="float32"):
        load_feature_matrix(tmp_path, "float64")

    out_of_range = np.zeros((7, 100), dtype=np.float32)
    out_of_range[0, 0] = 2.0
    _write_cache(tmp_path, "out-of-range", out_of_range)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        load_feature_matrix(tmp_path, "out-of-range")

    _write_cache(tmp_path, "missing-metadata", valid, metadata=False)
    with pytest.raises(FileNotFoundError, match="metadata"):
        load_feature_matrix(tmp_path, "missing-metadata")


def test_pairwise_dataset_uses_canonical_7_by_t_cache(tmp_path):
    matrix = np.zeros((7, 120), dtype=np.float32)
    matrix[0, 40:60] = 1.0
    _write_cache(tmp_path, "video", matrix)
    record = {
        "video_id": "video",
        "domain": "lecture",
        "source": "qvhighlights",
        "duration": 12.0,
        "relevant_windows": [[4.0, 6.0]],
        "saliency_scores": [],
    }

    pairs = create_pairwise_dataset(tmp_path, [record], window_sec=2.0, hop_sec=1.0)

    assert pairs
    assert pairs[0][0].shape == (7,)
    assert pairs[0][1].shape == (7,)
    assert pairs[0][0][0] > pairs[0][1][0]


def test_margin_ranking_loss_matches_hinge_formula():
    positive = torch.tensor([[2.0], [0.5]])
    negative = torch.tensor([[0.0], [0.25]])

    loss = margin_ranking_loss(positive, negative, gamma=1.0)

    assert float(loss) == pytest.approx((0.0 + 0.75) / 2)


def test_temporal_smoothness_is_per_video():
    constant = temporal_smoothness_loss([torch.tensor([1.0, 1.0, 1.0])])
    oscillating = temporal_smoothness_loss([torch.tensor([0.0, 1.0, 0.0])])
    separate_singletons = temporal_smoothness_loss([torch.tensor([0.0]), torch.tensor([10.0])])

    assert float(constant) == 0.0
    assert float(oscillating) == 1.0
    assert float(separate_singletons) == 0.0


def test_average_precision_uses_real_binary_metric():
    class FirstChannel(torch.nn.Module):
        def forward(self, features):
            return features[:, :1]

    examples = [
        _example("v1", 0, 0.1, 0),
        _example("v1", 1, 0.9, 1),
        _example("v1", 2, 0.2, 0),
        _example("v1", 3, 0.8, 1),
    ]

    assert evaluate_average_precision(FirstChannel(), examples) == pytest.approx(1.0)


def test_average_precision_rejects_single_class():
    model = torch.nn.Linear(7, 1)
    examples = [_example("v1", 0, 0.1, 1), _example("v1", 1, 0.2, 1)]

    with pytest.raises(ValueError, match="both positive and negative"):
        evaluate_average_precision(model, examples)


def test_train_writes_real_ap_checkpoint_and_log(tmp_path):
    train_matrix = np.zeros((7, 200), dtype=np.float32)
    train_matrix[0, 100:150] = 1.0
    val_matrix = np.zeros((7, 200), dtype=np.float32)
    val_matrix[0, 100:150] = 1.0
    _write_cache(tmp_path, "train-video", train_matrix)
    _write_cache(tmp_path, "val-video", val_matrix)
    output_path = tmp_path / "models" / "ltr.pt"
    log_path = tmp_path / "logs" / "training.json"

    model = train(
        tmp_path,
        [_qv_record("train-video")],
        output_path,
        [_qv_record("val-video")],
        max_epochs=3,
        patience=2,
        batch_size=4,
        lr=1e-2,
        seed=7,
        training_log_path=log_path,
    )

    checkpoint = torch.load(output_path, map_location="cpu", weights_only=False)
    metadata = checkpoint["metadata"]
    training_log = json.loads(log_path.read_text(encoding="utf-8"))
    assert isinstance(model, torch.nn.Module)
    assert metadata["selection_split"] == "validation"
    assert 0.0 <= metadata["val_ap"] <= 1.0
    assert metadata["feature_schema"]["channel_order"] == list(FEATURE_CHANNELS)
    assert metadata["dataset_fingerprint"]
    assert training_log["selection_split"] == "validation"
    assert training_log["epochs"]
    assert all("train_smooth_loss" in epoch for epoch in training_log["epochs"])
    assert all("val_ap" in epoch for epoch in training_log["epochs"])
    history_csv = log_path.with_name("training_history.csv")
    curves_svg = log_path.with_name("training_curves.svg")
    assert history_csv.is_file()
    assert curves_svg.is_file()
    history_text = history_csv.read_text(encoding="utf-8")
    assert "train_total_loss" in history_text
    assert "selection_ap" in history_text
    svg_text = curves_svg.read_text(encoding="utf-8")
    assert svg_text.startswith("<svg")
    ET.fromstring(svg_text)
    assert "Training losses" in svg_text
    assert f"Best epoch: {training_log['best_epoch']}" in svg_text
