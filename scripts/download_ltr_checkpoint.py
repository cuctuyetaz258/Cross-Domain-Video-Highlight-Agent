"""Download, verify, preflight, and atomically install the production LTR checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Callable

from highlight_agent.models.ltr_scorer import AdditiveAttentionScorer

DEFAULT_MANIFEST = Path("artifacts/manifests/ltr_scorer_v1_1.json")
DEFAULT_OUTPUT = Path("data/models/ltr_scorer.pt")


class CheckpointInstallError(RuntimeError):
    """A safe, actionable checkpoint installation failure."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CheckpointInstallError("checkpoint manifest must be a JSON object")
    required = {
        "artifact_version",
        "download_urls",
        "filename",
        "size_bytes",
        "sha256",
        "checkpoint_version",
        "feature_schema",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise CheckpointInstallError(
            f"checkpoint manifest is missing fields: {', '.join(missing)}"
        )
    urls = payload["download_urls"]
    if not isinstance(urls, list) or not urls or not all(isinstance(url, str) for url in urls):
        raise CheckpointInstallError("download_urls must be a non-empty string list")
    expected_hash = str(payload["sha256"]).lower()
    if len(expected_hash) != 64 or any(character not in "0123456789abcdef" for character in expected_hash):
        raise CheckpointInstallError("manifest sha256 must contain 64 hexadecimal characters")
    payload["sha256"] = expected_hash
    if int(payload["size_bytes"]) <= 0:
        raise CheckpointInstallError("manifest size_bytes must be positive")
    return payload


def _download(url: str, destination: Path, *, timeout_seconds: float) -> None:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Cross-Domain-Video-Highlight-Agent/1.1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
            with destination.open("wb") as handle:
                shutil.copyfileobj(response, handle)
    except urllib.error.HTTPError as exc:
        detail = (
            "; ensure the Kaggle dataset is published and Public"
            if exc.code in {401, 403, 404}
            else ""
        )
        raise CheckpointInstallError(
            f"checkpoint download returned HTTP {exc.code}{detail}"
        ) from exc
    except urllib.error.URLError as exc:
        raise CheckpointInstallError(f"checkpoint download failed: {exc.reason}") from exc


def _read_checkpoint_from_zip(archive: Path, filename: str, destination: Path) -> None:
    try:
        with zipfile.ZipFile(archive) as handle:
            matches = [
                item
                for item in handle.infolist()
                if not item.is_dir() and Path(item.filename).name == filename
            ]
            if len(matches) != 1:
                raise CheckpointInstallError(
                    f"download archive must contain exactly one {filename}; found {len(matches)}"
                )
            with handle.open(matches[0]) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)
    except zipfile.BadZipFile as exc:
        raise CheckpointInstallError("Kaggle response is not a valid ZIP archive") from exc


def _verify_checkpoint(path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    actual_size = path.stat().st_size
    expected_size = int(manifest["size_bytes"])
    if actual_size != expected_size:
        raise CheckpointInstallError(
            f"checkpoint size mismatch: got {actual_size}, expected {expected_size}"
        )
    actual_hash = _sha256(path)
    expected_hash = str(manifest["sha256"]).lower()
    if actual_hash != expected_hash:
        raise CheckpointInstallError(
            f"checkpoint SHA-256 mismatch: got {actual_hash}, expected {expected_hash}"
        )
    try:
        preflight = AdditiveAttentionScorer.preflight(path, device="cpu")
    except Exception as exc:
        raise CheckpointInstallError(f"checkpoint preflight failed: {exc}") from exc
    if preflight["checkpoint_version"] != manifest["checkpoint_version"]:
        raise CheckpointInstallError("checkpoint version does not match the artifact manifest")
    if preflight["feature_contract"]["schema_version"] != manifest["feature_schema"]:
        raise CheckpointInstallError("feature schema does not match the artifact manifest")
    expected_channels = manifest.get("channel_order")
    if expected_channels is not None and preflight["feature_contract"]["channel_order"] != expected_channels:
        raise CheckpointInstallError("channel order does not match the artifact manifest")
    expected_dataset = manifest.get("dataset_fingerprint")
    if expected_dataset is not None and preflight["dataset_fingerprint"] != expected_dataset:
        raise CheckpointInstallError("dataset fingerprint does not match the artifact manifest")
    return preflight


def install_checkpoint(
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    output_path: str | Path = DEFAULT_OUTPUT,
    force: bool = False,
    timeout_seconds: float = 120.0,
    downloader: Callable[[str, Path], None] | None = None,
) -> dict[str, Any]:
    """Install the exact manifest-bound checkpoint without exposing partial files."""

    manifest = load_manifest(manifest_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_file():
        if output.stat().st_size == int(manifest["size_bytes"]) and _sha256(output) == manifest["sha256"]:
            preflight = _verify_checkpoint(output, manifest)
            return {
                "status": "already_installed",
                "artifact_version": manifest["artifact_version"],
                "output": str(output.resolve()),
                "fingerprint": preflight["fingerprint"],
            }
        if not force:
            raise CheckpointInstallError(
                f"output already exists but does not match the manifest: {output}; use --force to replace it"
            )

    download = downloader or (
        lambda url, destination: _download(
            url,
            destination,
            timeout_seconds=timeout_seconds,
        )
    )
    with tempfile.TemporaryDirectory(prefix="ltr-checkpoint-", dir=output.parent) as temporary:
        temporary_dir = Path(temporary)
        archive = temporary_dir / "artifact.zip"
        checkpoint = temporary_dir / str(manifest["filename"])
        errors: list[str] = []
        for url in manifest["download_urls"]:
            try:
                download(url, archive)
                break
            except CheckpointInstallError as exc:
                errors.append(str(exc))
        else:
            raise CheckpointInstallError("all checkpoint download URLs failed: " + "; ".join(errors))
        _read_checkpoint_from_zip(archive, str(manifest["filename"]), checkpoint)
        preflight = _verify_checkpoint(checkpoint, manifest)
        os.replace(checkpoint, output)

    return {
        "status": "installed",
        "artifact_version": manifest["artifact_version"],
        "output": str(output.resolve()),
        "fingerprint": preflight["fingerprint"],
        "device_verified": preflight["device"],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    if args.timeout_seconds <= 0:
        raise CheckpointInstallError("timeout-seconds must be positive")
    result = install_checkpoint(
        manifest_path=args.manifest,
        output_path=args.output,
        force=args.force,
        timeout_seconds=args.timeout_seconds,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    main()
