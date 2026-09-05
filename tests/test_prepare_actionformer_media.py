import json

from scripts.prepare_actionformer_media import missing_video_ids


def test_missing_video_ids_reads_only_missing_artifact_issues(tmp_path) -> None:
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps(
            {
                "issues": [
                    {"video_id": "second", "issue": "missing artifacts: video"},
                    {"video_id": "ignored", "issue": "duration mismatch"},
                    {"video_id": "first", "issue": "missing artifacts: feature"},
                ]
            }
        ),
        encoding="utf-8",
    )

    assert missing_video_ids(audit) == ["first", "second"]
