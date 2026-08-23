"""Trích xuất feature thô Sprint 2 từ audio đã có trong workspace"""

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from highlight_agent.features import (
    build_feature_timeline,
    extract_interaction_features,
    extract_windowed_acoustic_features,
    save_feature_timeline,
    windowed_interaction_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio_path", type=Path, help="Path to the workspace WAV file")
    parser.add_argument("--domain", required=True, choices=["lecture", "podcast", "standup"])
    parser.add_argument("--video-id", default=None, help="Defaults to the audio parent folder name")
    parser.add_argument(
        "--known-speaker-count",
        type=int,
        default=None,
        help="Optional known number of Podcast speakers for Pyannote clustering",
    )
    parser.add_argument("--window-seconds", type=float, default=30.0)
    parser.add_argument("--hop-seconds", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.audio_path.is_file():
        raise FileNotFoundError(f"audio file does not exist: {args.audio_path}")

    load_dotenv()
    acoustic, acoustic_windows = extract_windowed_acoustic_features(
        args.audio_path,
        window_seconds=args.window_seconds,
        hop_seconds=args.hop_seconds,
    )
    interaction = None
    interaction_windows = None
    if args.domain == "podcast":
        interaction = extract_interaction_features(
            args.audio_path,
            num_speakers=args.known_speaker_count,
        )
        interaction_windows = windowed_interaction_features(
            interaction,
            window_seconds=args.window_seconds,
            hop_seconds=args.hop_seconds,
        )

    timeline = build_feature_timeline(
        video_id=args.video_id or args.audio_path.parent.name,
        domain=args.domain,
        duration=acoustic.duration,
        window_seconds=args.window_seconds,
        hop_seconds=args.hop_seconds,
        acoustic=acoustic,
        acoustic_windows=acoustic_windows,
        interaction=interaction,
        interaction_windows=interaction_windows,
    )
    output_path = save_feature_timeline(
        timeline,
        args.audio_path.parent / "features" / "features.json",
    )
    print(
        json.dumps(
            {
                "feature_path": str(output_path),
                "duration": timeline.duration,
                "window_count": len(timeline.windows),
                "speaker_count": interaction.speaker_count if interaction else None,
                "turn_count": interaction.turn_count if interaction else None,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
