from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from highlight_agent.features.visual_new import GestureExtraction
from scripts.build_feature_cache import refresh_gesture_observation_for_record, transcript_word_scores
from scripts.validate_training_data import resolve_record_path, validate_manifest


def test_manifest_paths_accept_both_separators_and_reject_machine_paths(tmp_path):
    expected = tmp_path / "data" / "raw" / "clip.mp4"
    assert resolve_record_path("data/raw/clip.mp4", tmp_path) == expected
    assert resolve_record_path("data\\raw\\clip.mp4", tmp_path) == expected

    for value in ("/tmp/clip.mp4", "C:\\media\\clip.mp4", "../clip.mp4"):
        try:
            resolve_record_path(value, tmp_path)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected {value!r} to be rejected")


def test_transcript_word_scores_are_aligned_and_normalized():
    transcript = SimpleNamespace(
        segments=[
            SimpleNamespace(
                text="common words",
                start=0.0,
                end=1.0,
                words=[SimpleNamespace(start=0.0, end=0.5)],
            ),
            SimpleNamespace(text="rare technical phrase", start=1.0, end=2.0, words=[]),
        ]
    )

    scores = transcript_word_scores(transcript)

    assert len(scores) == 2
    assert scores[0][:2] == (0.0, 0.5)
    assert scores[1][:2] == (1.0, 2.0)
    assert all(0.0 <= score <= 1.0 for _, _, score in scores)


def test_manifest_validator_reports_split_distribution(tmp_path, monkeypatch):
    for name in ("video.mp4", "audio.wav", "transcript.json"):
        (tmp_path / name).write_bytes(b"placeholder")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "video_id": "video",
                "source": "custom_pseudo",
                "domain": "lecture",
                "split": "train",
                "video_path": "video.mp4",
                "audio_path": "audio.wav",
                "transcript_path": "transcript.json",
                "duration": 20.0,
                "fps": 25.0,
                "relevant_windows": [[10.0, 15.0]],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("scripts.validate_training_data.probe_video", lambda path: (20.0, 25.0))
    monkeypatch.setattr(
        "scripts.validate_training_data.load_transcript",
        lambda path: SimpleNamespace(video_id="video", duration=20.0),
    )

    report = validate_manifest(manifest, project_root=tmp_path)

    assert report["valid"]
    assert report["splits"]["train"]["positive_windows"] > 0
    assert report["splits"]["train"]["negative_windows"] > 0
    assert report["splits"]["train"]["domains"] == {"lecture": 1}
    assert report["warnings"]


def test_manifest_validator_detects_video_split_leakage(tmp_path, monkeypatch):
    for name in ("video.mp4", "audio.wav", "transcript.json"):
        (tmp_path / name).write_bytes(b"placeholder")
    base = {
        "source": "custom_pseudo",
        "domain": "lecture",
        "video_path": "video.mp4",
        "audio_path": "audio.wav",
        "transcript_path": "transcript.json",
        "duration": 20.0,
        "fps": 25.0,
        "relevant_windows": [[10.0, 15.0]],
    }
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({**base, "video_id": "train-video", "split": "train"})
        + "\n"
        + json.dumps({**base, "video_id": "val-video", "split": "val"}),
        encoding="utf-8",
    )
    monkeypatch.setattr("scripts.validate_training_data.probe_video", lambda path: (20.0, 25.0))
    monkeypatch.setattr(
        "scripts.validate_training_data.load_transcript",
        lambda path: SimpleNamespace(video_id="train-video", duration=20.0),
    )

    report = validate_manifest(manifest, project_root=tmp_path)

    assert not report["valid"]
    assert any("duplicates" in error for error in report["errors"])


def test_refresh_gesture_observation_updates_only_metadata(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache" / "video"
    cache_dir.mkdir(parents=True)
    metadata_path = cache_dir / "metadata.json"
    metadata_path.write_text(json.dumps({"extractor": {}, "observations": {}}), encoding="utf-8")
    monkeypatch.setattr(
        "scripts.build_feature_cache.load_feature_matrix",
        lambda cache_root, video_id: np.zeros((7, 20), dtype=np.float32),
    )
    monkeypatch.setattr("scripts.build_feature_cache.resolve_record_path", lambda value, root: tmp_path / value)
    monkeypatch.setattr(
        "scripts.build_feature_cache.extract_gesture_observation",
        lambda path, duration, sample_rate: GestureExtraction(
            np.zeros(4, dtype=np.float32), "facemesh_initialization_failed", 0, 0
        ),
    )

    result = refresh_gesture_observation_for_record(
        {"video_id": "video", "video_path": "video.mp4", "duration": 2.0},
        project_root=tmp_path,
        output_dir=tmp_path / "cache",
    )

    updated = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert result["status"] == "refreshed_gesture_observation"
    assert updated["extractor"]["gesture_status"] == "facemesh_initialization_failed"
    assert updated["observations"]["gesture_detected_sample_count"] == 0
