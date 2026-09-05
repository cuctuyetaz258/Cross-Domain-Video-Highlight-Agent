from __future__ import annotations

from pathlib import Path

from scripts.build_kaggle_benchmark_media_bundle import stage_media


def test_stage_media_hardlinks_manifest_videos(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"media")
    output = tmp_path / "bundle"

    staged = stage_media(
        [{"video_id": "example", "source": "tvsum", "video_path": str(source)}], output
    )

    destination = output / "tvsum/videos/source.mp4"
    assert staged == [destination]
    assert destination.read_bytes() == b"media"
    assert destination.stat().st_ino == source.stat().st_ino
