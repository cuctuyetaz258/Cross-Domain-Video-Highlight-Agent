from pathlib import Path

import pytest

from highlight_agent.media import (
    InvalidVideoInputError,
    canonicalize_youtube_url,
    create_workspace,
    extract_youtube_id,
)


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://www.youtube.com/watch?v=jbL9kl4KPZI", "jbL9kl4KPZI"),
        ("https://youtu.be/jbL9kl4KPZI?t=30", "jbL9kl4KPZI"),
        ("https://www.youtube.com/shorts/jbL9kl4KPZI", "jbL9kl4KPZI"),
        ("https://music.youtube.com/watch?v=jbL9kl4KPZI&list=RDAMVM", "jbL9kl4KPZI"),
        ("https://www.youtube-nocookie.com/embed/jbL9kl4KPZI", "jbL9kl4KPZI"),
        ("https://www.youtube.com/watch/?v=jbL9kl4KPZI&si=tracking", "jbL9kl4KPZI"),
    ],
)
def test_extract_youtube_id(url: str, expected: str) -> None:
    assert extract_youtube_id(url) == expected


def test_canonicalize_youtube_url_removes_playlist_and_tracking_parameters() -> None:
    assert canonicalize_youtube_url(
        "https://youtu.be/jbL9kl4KPZI?si=tracking&t=30"
    ) == "https://www.youtube.com/watch?v=jbL9kl4KPZI"


def test_create_local_workspace_uses_local_source_without_copy(tmp_path: Path) -> None:
    video = tmp_path / "My lecture.mp4"
    video.write_bytes(b"fixture")

    workspace = create_workspace(str(video), tmp_path / "output")

    assert workspace.source_type == "local"
    assert workspace.source_video_path == video.resolve()
    assert workspace.video_id.startswith("My-lecture-")
    assert workspace.audio_path.parent.is_dir()
    assert (workspace.audio_path.parent / "shorts").is_dir()


def test_create_workspace_rejects_missing_local_file(tmp_path: Path) -> None:
    with pytest.raises(InvalidVideoInputError, match="does not exist"):
        create_workspace(str(tmp_path / "missing.mp4"), tmp_path / "output")
