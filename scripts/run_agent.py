"""Run the checkpoint-required LTR pipeline with optional LLM reranking."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Force UTF-8 output trên Windows (tránh cp1252 encode error với emoji)
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

try:
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table

    HAS_RICH = True
    console = Console(force_terminal=True)
except ImportError:
    HAS_RICH = False
    console = None

from highlight_agent.agent import build_agent_graph  # noqa: E402
from highlight_agent.agent.state import ProgressEvent  # noqa: E402
from highlight_agent.features.ltr_contract import LTRPipelineError  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
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
    parser.add_argument("--min-speaker-count", type=int, default=None)
    parser.add_argument("--max-speaker-count", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--cookies-browser", default=None)
    parser.add_argument(
        "--transcript-source",
        choices=["auto", "youtube", "whisper"],
        default="auto",
    )
    parser.add_argument(
        "--aspect-ratio",
        choices=["9:16", "16:9"],
        default="9:16",
        help="Tỷ lệ khung hình video xuất ra ('9:16' dọc hoặc '16:9' ngang).",
    )
    parser.add_argument("--no-subtitles", action="store_true")
    parser.add_argument(
        "--ltr-model-path",
        default="data/models/ltr_scorer.pt",
        help="Required compatible LTR checkpoint (.pt).",
    )
    parser.add_argument(
        "--llm-provider",
        choices=["auto", "groq", "openai", "custom", "disabled"],
        default="auto",
        help="Bật semantic reranking bằng LLM (mặc định 'auto' tự động bật khi có API key trong environment).",
    )
    parser.add_argument("--llm-model", default=None, help="Model slug; bỏ trống để dùng mặc định theo provider.")
    parser.add_argument("--llm-base-url", default=None, help="OpenAI-compatible base URL cho provider custom.")
    parser.add_argument("--llm-top-m", type=int, default=10, choices=range(3, 13))
    parser.add_argument("--llm-ltr-weight", type=float, default=0.60)
    parser.add_argument("--llm-timeout-seconds", type=float, default=45.0)
    args = parser.parse_args(argv)
    if args.known_speaker_count is not None and (
        args.min_speaker_count is not None or args.max_speaker_count is not None
    ):
        parser.error("--known-speaker-count cannot be combined with speaker count bounds")
    if args.min_speaker_count is not None and args.max_speaker_count is not None:
        if args.min_speaker_count > args.max_speaker_count:
            parser.error("--min-speaker-count cannot exceed --max-speaker-count")
    return args


def main() -> None:
    args = parse_args()

    rows: list[str] = []

    def render_panel():
        if not HAS_RICH:
            return ""
        table = Table(show_header=False, box=None, padding=(0, 1))
        for row in rows[-25:]:
            table.add_row(row)
        if not rows:
            table.add_row("[dim]Đang khởi tạo...[/dim]")
        return Panel(
            table,
            title="[bold blue]🎬 LTR-Required Highlight Agent[/bold blue]",
            border_style="blue",
        )

    live_context = None

    def emit(event: ProgressEvent):
        icon = {
            "start": "🔄",
            "preflight": "🔐",
            "visual_start": "🔄",
            "visual_window": "📐",
            "visual_normalize": "⚙️",
            "visual_done": "📊",
            "done": "✅",
            "fallback": "⚠️",
            "ranking": "🏆",
            "rendering": "🎞️",
            "llm_start": "🧠",
            "llm_done": "✅",
            "llm_fallback": "⚠️",
        }.get(event.step, "ℹ️")
        msg = f"{icon} [{event.node}/{event.step}] {event.message}"
        rows.append(msg)
        if HAS_RICH and live_context is not None:
            live_context.update(render_panel())
        else:
            print(msg, flush=True)

    state = {
        "video_path": args.video_input,
        "domain": args.domain,
        "highlight_count": args.highlight_count,
        "aspect_ratio": args.aspect_ratio,
        "known_speaker_count": args.known_speaker_count,
        "min_speaker_count": args.min_speaker_count,
        "max_speaker_count": args.max_speaker_count,
        "output_root": args.output_dir,
        "cookies_browser": args.cookies_browser,
        "transcript_source": args.transcript_source,
        "burn_subtitles": not args.no_subtitles,
        "ltr_model_path": args.ltr_model_path,
        "llm_provider": args.llm_provider,
        "llm_model": args.llm_model,
        "llm_base_url": args.llm_base_url,
        "llm_top_m": args.llm_top_m,
        "llm_ltr_weight": args.llm_ltr_weight,
        "llm_timeout_seconds": args.llm_timeout_seconds,
        "emit": emit,
    }
    graph = build_agent_graph()

    if HAS_RICH and console is not None:
        with Live(render_panel(), console=console, refresh_per_second=10) as live:
            live_context = live
            accumulated_state = dict(state)
            for step_output in graph.stream(state):
                node_name = list(step_output.keys())[0]
                accumulated_state.update(step_output[node_name])
        console.print("\n[bold green]✅ Pipeline hoàn tất![/bold green]\n")
        result = accumulated_state
    else:
        accumulated_state = dict(state)
        for step_output in graph.stream(state):
            node_name = list(step_output.keys())[0]
            accumulated_state.update(step_output[node_name])
        result = accumulated_state

    summary = {
        "workspace": result["workspace"].model_dump(mode="json"),
        "features": result["features"],
        "highlights": [item.model_dump(mode="json") for item in result.get("highlights", [])],
        "boundary_adjustments": [item.model_dump(mode="json") for item in result.get("boundary_adjustments", [])],
        "rendered_highlights": [item.model_dump(mode="json") for item in result.get("rendered_highlights", [])],
        "reasoning": result.get("reasoning", []),
        "llm_assessments": [
            item.model_dump(mode="json") for item in result.get("llm_assessments", [])
        ],
        "llm_run": (
            result["llm_run"].model_dump(mode="json") if result.get("llm_run") else None
        ),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def cli() -> None:
    """Run the CLI and serialize required-LTR failures for automation and UI wrappers."""

    try:
        main()
    except LTRPipelineError as exc:
        print(
            json.dumps(
                {"error": {"code": exc.code, "message": exc.message}},
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        raise SystemExit(2) from None


if __name__ == "__main__":
    cli()
