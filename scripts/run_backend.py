"""Chạy Backend Sprint 1 không qua LangGraph hoặc LLM"""

import argparse
import json

from highlight_agent.backend import load_candidates, prepare_video, render_candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video_input", help="YouTube URL or local video path")
    parser.add_argument("--candidates", help="Optional candidate JSON produced by an LLM/scoring step")
    parser.add_argument("--output-dir", default=None, help="Override OUTPUT_DIR")
    # parser.add_argument("--cookies-browser", default=None, help="Browser used by yt-dlp, e.g. chrome")
    parser.add_argument(
        "--transcript-source",
        choices=["auto", "youtube", "whisper"],
        default="auto",
        help="Choose caption-first, YouTube-only, or Whisper-only transcript",
    )
    parser.add_argument(
        "--aspect-ratio",
        choices=["9:16", "16:9"],
        default="9:16",
        help="Tỷ lệ khung hình video xuất ra ('9:16' dọc hoặc '16:9' ngang).",
    )
    parser.add_argument("--no-subtitles", action="store_true", help="Do not burn transcript subtitles")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace = prepare_video(
        args.video_input,
        output_root=args.output_dir,
        cookies_browser=args.cookies_browser,
        transcript_source=args.transcript_source,
    )
    print(json.dumps(workspace.model_dump(mode="json"), ensure_ascii=False, indent=2))

    if args.candidates:
        results = render_candidates(
            workspace,
            load_candidates(args.candidates),
            aspect_ratio=args.aspect_ratio,
            burn_subtitles=not args.no_subtitles,
        )
        print(json.dumps([item.model_dump(mode="json") for item in results], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
