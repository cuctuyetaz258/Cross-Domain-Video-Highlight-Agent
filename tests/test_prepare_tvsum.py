from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from highlight_agent.schemas import TranscriptDocument, TranscriptSegment
from scripts.prepare_tvsum import prepare_tvsum, select_tvsum_records
from scripts.validate_training_data import validate_manifest


def _records() -> list[dict]:
    return [
        {
            "video_id": video_id,
            "dataset": "tvsum",
            "category": category,
            "domain": "benchmark",
            "source": "tvsum",
            "fps": 10.0,
            "frame_scores": np.linspace(0, 1, 100, dtype=np.float32),
        }
        for video_id, category in [("alpha", "VT"), ("bravo", "VT"), ("charlie", "LF"), ("delta", "LF")]
    ]


def test_select_tvsum_records_is_video_split_deterministic() -> None:
    first = select_tvsum_records(_records(), limit=4, train_count=3, val_count=1, seed=42)
    second = select_tvsum_records(_records(), limit=4, train_count=3, val_count=1, seed=42)

    assert [(record["video_id"], record["category"], record["split"]) for record in first] == [
        (record["video_id"], record["category"], record["split"]) for record in second
    ]
    assert {record["domain"] for record in first} == {"benchmark"}
    assert {record["category"] for record in first} == {"LF", "VT"}
    assert sum(record["split"] == "train" for record in first) == 3
    assert sum(record["split"] == "val" for record in first) == 1
    assert len({record["video_id"] for record in first}) == 4


def test_prepare_tvsum_creates_manifest_with_raw_category(tmp_path: Path, monkeypatch) -> None:
    video_dir = tmp_path / "data" / "raw" / "tvsum" / "videos"
    video_dir.mkdir(parents=True)
    for video_id in ("alpha", "bravo"):
        (video_dir / f"{video_id}.mp4").write_bytes(b"video")
    annotations = tmp_path / "data" / "raw" / "tvsum" / "tvsum50.mat"
    annotations.write_bytes(b"annotation")

    monkeypatch.setattr("scripts.prepare_tvsum.load_tvsum", lambda _: _records()[:2])
    monkeypatch.setattr("scripts.prepare_tvsum.probe_duration", lambda _: 10.0)
    monkeypatch.setattr(
        "scripts.prepare_tvsum.extract_audio_16k_mono",
        lambda _, path: Path(path).parent.mkdir(parents=True, exist_ok=True) or Path(path).write_bytes(b"audio"),
    )

    def fake_transcript(audio_path, *, video_id, duration, model_size):
        assert Path(audio_path).is_file()
        assert model_size == "small.en"
        return TranscriptDocument(
            video_id=video_id,
            language="en",
            source="whisper",
            duration=duration,
            segments=[TranscriptSegment(id=0, start=0, end=1, text="Transcript")],
        )

    monkeypatch.setattr("scripts.prepare_tvsum.transcribe_with_whisper", fake_transcript)
    manifest = tmp_path / "data" / "manifests" / "tvsum_smoke.jsonl"
    report = prepare_tvsum(
        annotations_path=annotations,
        video_dir=video_dir,
        processed_dir=tmp_path / "data" / "raw" / "tvsum" / "processed",
        manifest_path=manifest,
        project_root=tmp_path,
        limit=2,
        train_count=1,
        val_count=1,
    )

    records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    assert report["prepared_count"] == 2
    assert {record["category"] for record in records} == {"VT"}
    assert {record["domain"] for record in records} == {"benchmark"}
    assert all((tmp_path / record["audio_path"]).is_file() for record in records)
    assert all((tmp_path / record["transcript_path"]).is_file() for record in records)


def test_prepare_tvsum_replaces_video_without_transcript(tmp_path: Path, monkeypatch) -> None:
    video_dir = tmp_path / "data" / "raw" / "tvsum" / "videos"
    video_dir.mkdir(parents=True)
    for record in _records():
        (video_dir / f"{record['video_id']}.mp4").write_bytes(b"video")
    annotations = tmp_path / "data" / "raw" / "tvsum" / "tvsum50.mat"
    annotations.write_bytes(b"annotation")

    monkeypatch.setattr("scripts.prepare_tvsum.load_tvsum", lambda _: _records())
    monkeypatch.setattr("scripts.prepare_tvsum.probe_duration", lambda _: 10.0)
    monkeypatch.setattr(
        "scripts.prepare_tvsum.extract_audio_16k_mono",
        lambda _, path: Path(path).parent.mkdir(parents=True, exist_ok=True) or Path(path).write_bytes(b"audio"),
    )
    calls = 0

    def fake_transcript(_audio_path, *, video_id, duration, model_size):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("Whisper did not detect usable English speech")
        return TranscriptDocument(
            video_id=video_id,
            language="en",
            source="whisper",
            duration=duration,
            segments=[TranscriptSegment(id=0, start=0, end=1, text="Transcript")],
        )

    monkeypatch.setattr("scripts.prepare_tvsum.transcribe_with_whisper", fake_transcript)
    manifest = tmp_path / "data" / "manifests" / "tvsum_smoke.jsonl"
    report = prepare_tvsum(
        annotations_path=annotations,
        video_dir=video_dir,
        processed_dir=tmp_path / "data" / "raw" / "tvsum" / "processed",
        manifest_path=manifest,
        project_root=tmp_path,
        limit=2,
        train_count=1,
        val_count=1,
    )

    assert report["ready"]
    assert report["prepared_count"] == 2
    assert any(result["status"] == "skipped_no_transcript" for result in report["results"])
    assert len(manifest.read_text(encoding="utf-8").splitlines()) == 2


def test_validator_accepts_tvsum_category_without_app_domain(tmp_path: Path, monkeypatch) -> None:
    for name in ("video.mp4", "audio.wav", "transcript.json"):
        (tmp_path / name).write_bytes(b"placeholder")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "video_id": "tvsum-video",
                "dataset": "tvsum",
                "category": "VT",
                "domain": "benchmark",
                "source": "tvsum",
                "split": "train",
                "video_path": "video.mp4",
                "audio_path": "audio.wav",
                "transcript_path": "transcript.json",
                "duration": 10.0,
                "fps": 10.0,
                "frame_scores": np.linspace(0, 1, 100).tolist(),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("scripts.validate_training_data.probe_video", lambda _: (10.0, 10.0))
    monkeypatch.setattr(
        "scripts.validate_training_data.load_transcript",
        lambda _: TranscriptDocument(
            video_id="tvsum-video",
            language="en",
            source="whisper",
            duration=10.0,
            segments=[TranscriptSegment(id=0, start=0, end=1, text="Transcript")],
        ),
    )

    report = validate_manifest(manifest, project_root=tmp_path)

    assert report["valid"]
    assert report["splits"]["train"]["domains"] == {"benchmark": 1}
