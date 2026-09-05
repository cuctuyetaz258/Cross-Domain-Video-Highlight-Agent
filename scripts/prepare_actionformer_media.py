"""Download missing ActionFormer pilot media and persist resumable progress."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from highlight_agent.media import prepare_media_workspace


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def missing_video_ids(audit_path: str | Path) -> list[str]:
    audit = json.loads(Path(audit_path).read_text(encoding="utf-8"))
    return sorted(
        str(item["video_id"])
        for item in audit.get("issues", [])
        if str(item.get("issue", "")).startswith("missing artifacts")
    )


def prepare_missing_media(
    video_ids: list[str],
    *,
    output_root: str | Path,
    report_path: str | Path,
    cookies_browser: str | None = None,
    transcript_source: str = "auto",
) -> dict[str, Any]:
    root = Path(output_root)
    report_file = Path(report_path)
    started = time.time()
    results: list[dict[str, Any]] = []
    report: dict[str, Any] = {}
    for index, video_id in enumerate(video_ids, start=1):
        workspace_dir = root / video_id
        required = [
            workspace_dir / "source_video.mp4",
            workspace_dir / "audio.wav",
            workspace_dir / "transcript.json",
        ]
        item_started = time.perf_counter()
        print(f"[{index}/{len(video_ids)}] preparing {video_id}...", flush=True)
        if all(path.is_file() for path in required):
            result: dict[str, Any] = {
                "video_id": video_id,
                "status": "skipped_ready",
            }
        else:
            try:
                workspace = prepare_media_workspace(
                    f"https://www.youtube.com/watch?v={video_id}",
                    output_root=root,
                    cookies_browser=cookies_browser,
                    transcript_source=transcript_source,
                )
                result = {
                    "video_id": video_id,
                    "status": "prepared",
                    "has_source_transcript": workspace.has_source_transcript,
                    "video_path": str(workspace.source_video_path),
                    "audio_path": str(workspace.audio_path),
                    "transcript_path": str(workspace.transcript_path),
                    "video_bytes": workspace.source_video_path.stat().st_size,
                    "audio_bytes": workspace.audio_path.stat().st_size,
                }
            except Exception as exc:  # noqa: BLE001 - batch must continue after one URL fails
                result = {
                    "video_id": video_id,
                    "status": "failed",
                    "error": str(exc),
                }
        result["elapsed_seconds"] = round(time.perf_counter() - item_started, 3)
        results.append(result)
        report = {
            "schema_version": "1.0",
            "status": "running",
            "output_root": str(root.resolve()),
            "requested_video_count": len(video_ids),
            "prepared_count": sum(item["status"] == "prepared" for item in results),
            "skipped_count": sum(item["status"] == "skipped_ready" for item in results),
            "failed_count": sum(item["status"] == "failed" for item in results),
            "started_at_unix": started,
            "updated_at_unix": time.time(),
            "results": results,
        }
        _write_report(report_file, report)
        print(json.dumps(result, sort_keys=True), flush=True)
    report["status"] = "complete"
    report["completed_at_unix"] = time.time()
    report["elapsed_seconds"] = report["completed_at_unix"] - started
    _write_report(report_file, report)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", default="data/reports/actionformer_data_audit.json")
    parser.add_argument("--output-root", default="data/raw/in_domain_pilot")
    parser.add_argument(
        "--report",
        default="data/reports/actionformer_media_prepare.json",
    )
    parser.add_argument("--cookies-browser", default=None)
    parser.add_argument(
        "--transcript-source",
        choices=["auto", "youtube", "whisper"],
        default="auto",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("video_ids", nargs="*")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    video_ids = args.video_ids or missing_video_ids(args.audit)
    if args.limit is not None:
        video_ids = video_ids[: args.limit]
    report = prepare_missing_media(
        video_ids,
        output_root=args.output_root,
        report_path=args.report,
        cookies_browser=args.cookies_browser,
        transcript_source=args.transcript_source,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if report["failed_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
