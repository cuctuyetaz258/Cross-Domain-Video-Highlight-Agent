import json

from highlight_agent.models.actionformer import TemporalProposal
from highlight_agent.models.oof_proposals import (
    OOF_PROPOSAL_CACHE_VERSION,
    load_oof_proposal_cache,
)


def test_load_oof_proposal_cache_round_trip(tmp_path) -> None:
    path = tmp_path / "oof.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": OOF_PROPOSAL_CACHE_VERSION,
                "status": "complete",
                "folds": [{"fold": 0}],
                "videos": {
                    "video": {
                        "proposals": [
                            {
                                "start": 1.0,
                                "end": 31.0,
                                "confidence": 0.75,
                                "level": 1,
                                "center_index": 4,
                            }
                        ]
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    proposals, metadata = load_oof_proposal_cache(path)

    assert proposals == {"video": [TemporalProposal(1.0, 31.0, 0.75, 1, 4)]}
    assert metadata["fold_count"] == 1
    assert metadata["video_count"] == 1


def test_load_oof_proposal_cache_rejects_incomplete_file(tmp_path) -> None:
    path = tmp_path / "oof.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": OOF_PROPOSAL_CACHE_VERSION,
                "status": "running",
                "videos": {},
            }
        ),
        encoding="utf-8",
    )

    try:
        load_oof_proposal_cache(path)
    except ValueError as error:
        assert "incomplete" in str(error)
    else:
        raise AssertionError("incomplete cache must be rejected")
