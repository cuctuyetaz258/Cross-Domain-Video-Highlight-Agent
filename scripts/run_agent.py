"""Chạy LangGraph năm pha — Sprint 2: Visual Scoring + Live Progress"""

import argparse
import json
import os
import sys

# Force UTF-8 output trên Windows (tránh cp1252 encode error với emoji)
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.live import Live

from highlight_agent.agent import build_agent_graph
from highlight_agent.agent.state import ProgressEvent
from highlight_agent.backend import load_candidates

console = Console(force_terminal=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video_input", help="YouTube URL or local video path")
    parser.add_argument("--domain", required=True, choices=["lecture", "podcast", "standup"])
    parser.add_argument("--highlight-count", type=int, default=3, choices=range(3, 6))
    parser.add_argument("--candidates", help="Optional external candidate JSON")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--cookies-browser", default=None)
    parser.add_argument(
        "--transcript-source",
        choices=["auto", "youtube", "whisper"],
        default="auto",
    )
    parser.add_argument("--no-subtitles", action="store_true")
    parser.add_argument(
        "--visual-method",
        choices=["pixel_diff", "raft"],
        default="pixel_diff",
        help="Phương pháp tính visual score. Dùng 'raft' nếu có GPU.",
    )
    parser.add_argument(
        "--visual-sample-fps",
        type=float,
        default=1.0,
        help="Số frame lấy mẫu mỗi giây (1.0 = nhanh, 2.0 = chính xác hơn).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rows: list[str] = []

    def render_panel():
        table = Table(show_header=False, box=None, padding=(0, 1))
        for row in rows[-25:]:  # hiện 25 dòng cuối
            table.add_row(row)
        if not rows:
            table.add_row("[dim]Đang khởi tạo...[/dim]")
        return Panel(
            table,
            title="[bold blue]🎬 Agent Flow[/bold blue]",
            border_style="blue",
        )

    def emit(event: ProgressEvent):
        icon = {
            "start": "🔄",
            "visual_start": "🔄",
            "visual_window": "📐",
            "visual_normalize": "⚙️",
            "visual_done": "📊",
            "done": "✅",
            "fallback": "⚠️",
            "ranking": "🏆",
        }.get(event.step, "ℹ️")
        rows.append(f"{icon} [bold]{event.node}[/bold]/{event.step} │ {event.message}")
        live.update(render_panel())

    state = {
        "video_path": args.video_input,
        "domain": args.domain,
        "highlight_count": args.highlight_count,
        "output_root": args.output_dir,
        "cookies_browser": args.cookies_browser,
        "transcript_source": args.transcript_source,
        "burn_subtitles": not args.no_subtitles,
        "visual_method": args.visual_method,
        "visual_sample_fps": args.visual_sample_fps,
        "emit": emit,
    }
    if args.candidates:
        state["candidates"] = load_candidates(args.candidates)

    graph = build_agent_graph()

    with Live(render_panel(), console=console, refresh_per_second=10) as live:
        accumulated_state = dict(state)
        for step_output in graph.stream(state):
            node_name = list(step_output.keys())[0]
            accumulated_state.update(step_output[node_name])

    console.print("\n[bold green]✅ Pipeline hoàn tất![/bold green]\n")

    # Output JSON summary
    result = accumulated_state
    summary = {
        "workspace": result["workspace"].model_dump(mode="json"),
        "features": result["features"],
        "highlights": [item.model_dump(mode="json") for item in result["highlights"]],
        "rendered_highlights": [
            item.model_dump(mode="json") for item in result["rendered_highlights"]
        ],
        "reasoning": result["reasoning"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
