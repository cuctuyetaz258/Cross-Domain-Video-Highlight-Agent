"""Chạy LangGraph năm pha cho baseline Sprint 1"""

import argparse
import json

from highlight_agent.agent import build_agent_graph
from highlight_agent.backend import load_candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video_input", help="YouTube URL or local video path")
    parser.add_argument("--domain", required=True, choices=["lecture", "podcast", "standup"])
    parser.add_argument("--highlight-count", type=int, default=3, choices=range(3, 6))
    parser.add_argument(
        "--known-speaker-count",
        type=int,
        default=None,
        help="Số speaker đã biết, dùng cho Pyannote diarization của podcast",
    )
    parser.add_argument("--candidates", help="Optional external candidate JSON")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--cookies-browser", default=None)
    parser.add_argument(
        "--transcript-source",
        choices=["auto", "youtube", "whisper"],
        default="auto",
    )
    parser.add_argument("--no-subtitles", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state = {
        "video_path": args.video_input,
        "domain": args.domain,
        "highlight_count": args.highlight_count,
        "known_speaker_count": args.known_speaker_count,
        "output_root": args.output_dir,
        "cookies_browser": args.cookies_browser,
        "transcript_source": args.transcript_source,
        "burn_subtitles": not args.no_subtitles,
    }
    if args.candidates:
        state["candidates"] = load_candidates(args.candidates)

    result = build_agent_graph().invoke(state)
    summary = {
        "workspace": result["workspace"].model_dump(mode="json"),
        "features": result["features"],
        "highlights": [item.model_dump(mode="json") for item in result["highlights"]],
        "boundary_adjustments": [item.model_dump(mode="json") for item in result["boundary_adjustments"]],
        "rendered_highlights": [item.model_dump(mode="json") for item in result["rendered_highlights"]],
        "reasoning": result["reasoning"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
