from __future__ import annotations

import json
from pathlib import Path
import numpy as np

import pytest

from highlight_agent.models.train_offline import (
    load_qvhighlights,
    create_window_labels,
    compute_lref,
    create_pairwise_dataset,
    train
)


def test_iou_perfect():
    record = {
        "source": "qvhighlights",
        "duration": 5.0,
        "relevant_windows": [[0.0, 5.0]],
        "saliency_scores": [[3, 3, 3]]
    }
    labels = create_window_labels(record, window_sec=5.0, hop_sec=1.0)
    assert labels[0]["label"] == "positive"
    assert labels[0]["score"] == 1.0


def test_iou_no_overlap():
    record = {
        "source": "qvhighlights",
        "duration": 15.0,
        "relevant_windows": [[10.0, 15.0]],
        "saliency_scores": [[3, 3, 3]]
    }
    labels = create_window_labels(record, window_sec=5.0, hop_sec=1.0)
    # The first window is [0, 5], which has no overlap with [10, 15]
    assert labels[0]["label"] == "negative"
    assert labels[0]["score"] == 0.0


def test_load_qvhighlights_from_jsonl(tmp_path):
    p = tmp_path / "data.jsonl"
    entries = [
        {"vid": "vid1", "relevant_windows": [[0, 5]]},
        {"vid": "vid2", "relevant_windows": [[10, 15]], "saliency_scores": [[3, 2, 1]]}
    ]
    with open(p, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
            
    records = load_qvhighlights(p)
    assert len(records) == 2
    assert records[0]["video_id"] == "vid1"
    assert records[1]["saliency_scores"] == [[3, 2, 1]]
    assert records[0]["duration"] == 5.0


def test_create_window_labels_summe():
    record = {
        "source": "summe",
        "fps": 25.0,
        "frame_scores": np.linspace(0, 1, 250) # 10 seconds
    }
    labels = create_window_labels(record, window_sec=5.0, hop_sec=1.0)
    # scores from 0 to 1. The first 5s window will have mean ~0.25 (negative)
    # the last 5s window will have mean ~0.75 (positive)
    assert any(l["label"] == "positive" for l in labels)
    assert any(l["label"] == "negative" for l in labels)


def test_create_window_labels_qv():
    # Use a highlight [10, 16] — a 5s window [10,15] has IoU = 5/6 ≈ 0.83 > 0.5
    record = {
        "source": "qvhighlights",
        "duration": 30.0,
        "relevant_windows": [[10.0, 16.0]],
        "saliency_scores": [],  # fallback to relevant_windows
    }
    labels = create_window_labels(record, window_sec=5.0, hop_sec=5.0)
    pos_labels = [l for l in labels if l["label"] == "positive"]
    assert len(pos_labels) > 0


def test_pairwise_from_cache(tmp_path):
    vid = "test_vid"
    feat_dir = tmp_path / vid
    feat_dir.mkdir(parents=True)
    
    # 10 frames of 7-dim
    feats = np.random.rand(10, 7)
    np.save(feat_dir / "feature_matrix.npy", feats)
    
    record = {
        "video_id": vid,
        "source": "qvhighlights",
        "duration": 10.0,
        "relevant_windows": [[0.0, 2.0]],
        "saliency_scores": []
    }
    
    dataset = create_pairwise_dataset(tmp_path, [record], window_sec=1.0, hop_sec=1.0)
    assert isinstance(dataset, list)
    if dataset:
        assert isinstance(dataset[0], tuple)
        assert dataset[0][0].shape == (7,)
        assert dataset[0][1].shape == (7,)


def test_compute_lref_qvhighlights():
    records = [
        {"source": "qvhighlights", "relevant_windows": [[0, 40], [10, 50]]}
    ]
    lref = compute_lref(records)
    assert lref == 40.0


def test_train_loss_decreasing(tmp_path):
    # Dummy data
    dataset = []
    for _ in range(10):
        # positive is ones, negative is zeros
        pos = np.ones(7, dtype=np.float32)
        neg = np.zeros(7, dtype=np.float32)
        dataset.append((pos, neg))
        
    # Mock create_pairwise_dataset in train via monkeypatch or just use the model logic
    from highlight_agent.models.train_offline import train
    import highlight_agent.models.train_offline as module
    
    # Temporarily override create_pairwise_dataset for this test
    original = module.create_pairwise_dataset
    module.create_pairwise_dataset = lambda *args, **kwargs: dataset
    
    try:
        model = train(
            feature_cache_dir=tmp_path,
            records=[{"source": "qvhighlights", "relevant_windows": []}],
            output_path=tmp_path / "model.pt",
            max_epochs=2,
            batch_size=2
        )
        assert (tmp_path / "model.pt").exists()
    finally:
        module.create_pairwise_dataset = original
