from __future__ import annotations

import hashlib
import json
import zipfile

import pytest

from highlight_agent.features.ltr_contract import feature_contract
from highlight_agent.models.ltr_scorer import AdditiveAttentionScorer
from scripts.download_ltr_checkpoint import CheckpointInstallError, install_checkpoint


def _artifact_fixture(tmp_path):
    checkpoint = tmp_path / "source" / "ltr_scorer.pt"
    checkpoint.parent.mkdir()
    AdditiveAttentionScorer().save(
        checkpoint,
        metadata={
            "schema_version": "1.1",
            "feature_schema": feature_contract(),
            "L_ref": 30.0,
            "dataset_fingerprint": "fixture-dataset",
        },
    )
    archive = tmp_path / "artifact.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.write(checkpoint, "release/ltr_scorer.pt")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "artifact_version": "test-v1.1",
                "download_urls": ["https://example.invalid/checkpoint"],
                "filename": "ltr_scorer.pt",
                "size_bytes": checkpoint.stat().st_size,
                "sha256": digest,
                "checkpoint_version": "1.0",
                "feature_schema": "1.1",
            }
        ),
        encoding="utf-8",
    )
    return checkpoint, archive, manifest, digest


def test_installer_downloads_verifies_and_atomically_installs(tmp_path) -> None:
    _checkpoint, archive, manifest, digest = _artifact_fixture(tmp_path)
    output = tmp_path / "models" / "ltr_scorer.pt"

    result = install_checkpoint(
        manifest_path=manifest,
        output_path=output,
        downloader=lambda _url, destination: destination.write_bytes(archive.read_bytes()),
    )

    assert result["status"] == "installed"
    assert result["artifact_version"] == "test-v1.1"
    assert result["fingerprint"] == digest
    assert output.is_file()
    assert hashlib.sha256(output.read_bytes()).hexdigest() == digest
    assert not list(output.parent.glob("ltr-checkpoint-*"))


def test_installer_skips_download_for_valid_existing_checkpoint(tmp_path) -> None:
    checkpoint, _archive, manifest, digest = _artifact_fixture(tmp_path)
    output = tmp_path / "models" / "ltr_scorer.pt"
    output.parent.mkdir()
    output.write_bytes(checkpoint.read_bytes())

    result = install_checkpoint(
        manifest_path=manifest,
        output_path=output,
        downloader=lambda *_: pytest.fail("valid checkpoint must not be downloaded again"),
    )

    assert result["status"] == "already_installed"
    assert result["fingerprint"] == digest


def test_installer_does_not_replace_unexpected_existing_file_without_force(tmp_path) -> None:
    _checkpoint, archive, manifest, _digest = _artifact_fixture(tmp_path)
    output = tmp_path / "models" / "ltr_scorer.pt"
    output.parent.mkdir()
    output.write_bytes(b"user checkpoint")

    with pytest.raises(CheckpointInstallError, match="use --force"):
        install_checkpoint(
            manifest_path=manifest,
            output_path=output,
            downloader=lambda _url, destination: destination.write_bytes(archive.read_bytes()),
        )

    assert output.read_bytes() == b"user checkpoint"


def test_installer_keeps_existing_file_when_download_hash_is_wrong(tmp_path) -> None:
    _checkpoint, _archive, manifest, _digest = _artifact_fixture(tmp_path)
    output = tmp_path / "models" / "ltr_scorer.pt"
    output.parent.mkdir()
    output.write_bytes(b"existing")
    corrupt_archive = tmp_path / "corrupt.zip"
    corrupt_checkpoint = tmp_path / "wrong.pt"
    corrupt_checkpoint.write_bytes(b"wrong checkpoint")
    with zipfile.ZipFile(corrupt_archive, "w") as handle:
        handle.write(corrupt_checkpoint, "ltr_scorer.pt")

    with pytest.raises(CheckpointInstallError, match="size mismatch"):
        install_checkpoint(
            manifest_path=manifest,
            output_path=output,
            force=True,
            downloader=lambda _url, destination: destination.write_bytes(
                corrupt_archive.read_bytes()
            ),
        )

    assert output.read_bytes() == b"existing"
