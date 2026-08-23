"""Tạo trang review độc lập trên trình duyệt từ feature timeline Sprint 2"""

import argparse
import html
import json
import os
from pathlib import Path

from highlight_agent.schemas import FeatureTimeline


def _format_time(seconds: float) -> str:
    total_seconds = round(seconds)
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"


def _speaker_color(speaker: str) -> str:
    return "speaker-a" if speaker.endswith("00") else "speaker-b"


def build_review_html(timeline: FeatureTimeline, *, video_source: str) -> str:
    """Tạo trang review tự chứa, video vẫn được tham chiếu bằng đường dẫn tương đối"""

    interaction = timeline.interaction
    summary = {
        "duration": _format_time(timeline.duration),
        "windows": len(timeline.windows),
        "speakers": interaction.speaker_count if interaction else 0,
        "turns": interaction.turn_count if interaction else 0,
        "speech_ratio": round((interaction.speech_ratio if interaction else 0) * 100),
    }
    payload = {
        "duration": timeline.duration,
        "windows": [
            {
                "start": window.start,
                "end": window.end,
                "rms": window.acoustic.rms_mean,
                "silence": window.acoustic.silence_ratio,
                "pitch": window.acoustic.pitch_mean_hz,
                "interaction": window.interaction.model_dump(mode="json") if window.interaction else None,
            }
            for window in timeline.windows
        ],
        "turns": [turn.model_dump(mode="json") for turn in interaction.turns] if interaction else [],
    }
    payload_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    title = f"Diarization review - {timeline.video_id}"
    return f"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --ink: #1f2a25;
      --muted: #617069;
      --paper: #f6f2e9;
      --surface: #fffdf8;
      --line: #d9d4c8;
      --accent: #e5573a;
      --speaker-a: #2364aa;
      --speaker-b: #dd8b18;
      --quiet: #b9c4bd;
      --focus: #162d22;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background: radial-gradient(circle at 85% 0%, #dfe9d4 0, transparent 29rem), var(--paper);
      font-family: Georgia, "Times New Roman", serif;
    }}
    .page {{ max-width: 1180px; margin: 0 auto; padding: 48px 24px 72px; }}
    .eyebrow {{ color: var(--accent); font: 700 0.75rem/1.2 ui-monospace, SFMono-Regular, Menlo, monospace; letter-spacing: .13em; text-transform: uppercase; }}
    h1 {{ max-width: 15ch; margin: 10px 0 12px; font-size: clamp(2.35rem, 6vw, 4.8rem); line-height: .95; letter-spacing: -.055em; }}
    .intro {{ max-width: 58ch; color: var(--muted); font: 1.08rem/1.6 ui-sans-serif, system-ui, sans-serif; }}
    .facts {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; margin: 34px 0; }}
    .fact {{ min-height: 98px; padding: 16px; border: 1px solid var(--line); background: color-mix(in srgb, var(--surface) 85%, transparent); }}
    .fact strong {{ display: block; margin-top: 7px; font: 700 clamp(1.35rem, 3vw, 2.25rem)/1 ui-monospace, SFMono-Regular, Menlo, monospace; letter-spacing: -.08em; }}
    .fact span {{ color: var(--muted); font: .72rem/1.25 ui-monospace, SFMono-Regular, Menlo, monospace; text-transform: uppercase; letter-spacing: .06em; }}
    .panel {{ margin-top: 20px; padding: clamp(18px, 3vw, 32px); border: 1px solid var(--line); background: var(--surface); box-shadow: 8px 8px 0 rgba(31, 42, 37, .06); }}
    .section-head {{ display: flex; align-items: baseline; justify-content: space-between; gap: 16px; margin-bottom: 15px; }}
    h2 {{ margin: 0; font-size: clamp(1.35rem, 3vw, 2rem); letter-spacing: -.03em; }}
    .hint {{ margin: 0; color: var(--muted); font: .82rem/1.4 ui-sans-serif, system-ui, sans-serif; }}
    video {{ display: block; width: 100%; max-height: 68vh; min-height: 220px; background: #101612; accent-color: var(--accent); object-fit: contain; }}
    .now {{ min-height: 1.6em; margin: 12px 0 0; color: var(--focus); font: 700 .85rem/1.4 ui-monospace, SFMono-Regular, Menlo, monospace; }}
    .overview {{ position: relative; height: 82px; margin-top: 26px; overflow: hidden; border: 1px solid var(--ink); background: repeating-linear-gradient(90deg, transparent 0, transparent calc(4.142857% - 1px), rgba(31,42,37,.12) calc(4.142857% - 1px), rgba(31,42,37,.12) 4.142857%); }}
    .turn {{ position: absolute; top: 16px; height: 48px; min-width: 2px; border-radius: 2px; opacity: .95; }}
    .speaker-a {{ background: var(--speaker-a); }} .speaker-b {{ background: var(--speaker-b); }}
    .playhead {{ position: absolute; z-index: 2; top: 0; bottom: 0; width: 2px; background: var(--accent); transform: translateX(-1px); pointer-events: none; }}
    .legend {{ display: flex; flex-wrap: wrap; gap: 16px; margin: 12px 0 0; color: var(--muted); font: .78rem/1.4 ui-sans-serif, system-ui, sans-serif; }}
    .legend i {{ display: inline-block; width: 11px; height: 11px; margin-right: 6px; vertical-align: -1px; border-radius: 50%; }}
    .windows {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; }}
    .window {{ width: 100%; min-height: 118px; padding: 14px; cursor: pointer; border: 1px solid var(--line); border-left: 5px solid var(--quiet); color: var(--ink); background: var(--surface); text-align: left; transition: transform 180ms ease-out, border-color 180ms ease-out, background 180ms ease-out; }}
    .window:hover {{ transform: translateY(-2px); border-color: var(--ink); }}
    .window:focus-visible {{ outline: 3px solid var(--accent); outline-offset: 2px; }}
    .window.active {{ border-color: var(--focus); border-left-color: var(--accent); background: #eff3e8; }}
    .window-time {{ display: block; font: 700 .82rem/1.2 ui-monospace, SFMono-Regular, Menlo, monospace; }}
    .window-main {{ display: block; margin-top: 18px; font: 700 1.05rem/1.15 ui-sans-serif, system-ui, sans-serif; }}
    .window-detail {{ display: block; margin-top: 5px; color: var(--muted); font: .76rem/1.35 ui-monospace, SFMono-Regular, Menlo, monospace; }}
    .empty {{ padding: 20px; border: 1px dashed var(--line); color: var(--muted); font: 1rem/1.5 ui-sans-serif, system-ui, sans-serif; }}
    footer {{ margin-top: 24px; color: var(--muted); font: .78rem/1.5 ui-sans-serif, system-ui, sans-serif; }}
    @media (max-width: 720px) {{ .page {{ padding: 30px 16px 48px; }} .facts {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} .windows {{ grid-template-columns: 1fr; }} .section-head {{ align-items: flex-start; flex-direction: column; }} }}
    @media (prefers-reduced-motion: reduce) {{ * {{ scroll-behavior: auto !important; transition: none !important; }} }}
  </style>
</head>
<body>
  <main class="page">
    <p class="eyebrow">Highlight Agent / Sprint 2 / Human review</p>
    <h1>Ai đang nói, và model đổi người nói lúc nào?</h1>
    <p class="intro">Trang này đọc kết quả diarization đã có. Bấm một ô 30 giây để nhảy video đến đúng đoạn, rồi vừa xem vừa nghe xem màu speaker có khớp với cuộc trò chuyện không.</p>
    <section class="facts" aria-label="Tóm tắt phân tích">
      <div class="fact"><span>Thời lượng</span><strong>{summary["duration"]}</strong></div>
      <div class="fact"><span>Cửa sổ</span><strong>{summary["windows"]} × 30s</strong></div>
      <div class="fact"><span>Speakers</span><strong>{summary["speakers"]}</strong></div>
      <div class="fact"><span>Đổi speaker</span><strong>{summary["turns"]}</strong></div>
    </section>
    <section class="panel" aria-labelledby="listen-heading">
      <div class="section-head"><h2 id="listen-heading">Xem, nghe và theo dõi timeline</h2><p class="hint">Màu xanh / cam là nhãn tạm của model, không phải tên người thật.</p></div>
      <video id="video" controls preload="metadata"><source src="{html.escape(video_source, quote=True)}" type="video/mp4">Trình duyệt không hỗ trợ video MP4.</video>
      <p id="now" class="now" aria-live="polite">Chọn một cửa sổ để bắt đầu review.</p>
      <div id="overview" class="overview" role="img" aria-label="Timeline speaker diarization"><div id="playhead" class="playhead"></div></div>
      <div class="legend"><span><i class="speaker-a"></i>SPEAKER_00</span><span><i class="speaker-b"></i>SPEAKER_01</span><span>Speech detected: {summary["speech_ratio"]}%</span></div>
    </section>
    <section class="panel" aria-labelledby="windows-heading">
      <div class="section-head"><h2 id="windows-heading">Review theo cửa sổ 30 giây</h2><p class="hint">Bấm hoặc Tab + Enter để nhảy đến window. Không có turn change không có nghĩa là không có speech.</p></div>
      <div id="windows" class="windows"></div>
    </section>
    <footer>Generated from <code>features.json</code>. Dùng trang này để kiểm tra chất lượng diarization; đừng xem label speaker là danh tính đã được xác minh.</footer>
  </main>
  <script>
    const data = {payload_json};
    const player = document.getElementById("video");
    const overview = document.getElementById("overview");
    const playhead = document.getElementById("playhead");
    const now = document.getElementById("now");
    const windows = document.getElementById("windows");
    const formatTime = (seconds) => {{
      const total = Math.round(seconds);
      const minutes = Math.floor(total / 60);
      return `${{minutes}}:${{String(total % 60).padStart(2, "0")}}`;
    }};
    const setActive = (seconds) => {{
      const active = data.windows.find((item) => seconds >= item.start && seconds < item.end);
      document.querySelectorAll(".window").forEach((button) => button.classList.toggle("active", button.dataset.start === String(active?.start)));
      playhead.style.left = `${{Math.min(100, (seconds / data.duration) * 100)}}%`;
      now.textContent = `Đang nghe ${{formatTime(seconds)}}${{active ? ` · window ${{formatTime(active.start)}}–${{formatTime(active.end)}}` : ""}}`;
    }};
    data.turns.forEach((turn) => {{
      const segment = document.createElement("span");
      segment.className = `turn ${{turn.speaker.endsWith("00") ? "speaker-a" : "speaker-b"}}`;
      segment.style.left = `${{(turn.start / data.duration) * 100}}%`;
      segment.style.width = `${{Math.max(.2, ((turn.end - turn.start) / data.duration) * 100)}}%`;
      segment.title = `${{turn.speaker}}: ${{formatTime(turn.start)}}–${{formatTime(turn.end)}}`;
      overview.append(segment);
    }});
    data.windows.forEach((item) => {{
      const button = document.createElement("button");
      button.type = "button";
      button.className = "window";
      button.dataset.start = String(item.start);
      const interaction = item.interaction;
      const turns = interaction ? interaction.turn_count : 0;
      const speakers = interaction ? interaction.speaker_count : 0;
      button.setAttribute("aria-label", `Nghe từ ${{formatTime(item.start)}} đến ${{formatTime(item.end)}}`);
      button.innerHTML = `<span class="window-time">${{formatTime(item.start)}} – ${{formatTime(item.end)}}</span><span class="window-main">${{turns}} lượt đổi speaker</span><span class="window-detail">${{speakers}} speaker · silence ${{Math.round(item.silence * 100)}}% · RMS ${{item.rms.toFixed(3)}}</span>`;
      button.addEventListener("click", () => {{ player.currentTime = item.start; player.play(); setActive(item.start); }});
      windows.append(button);
    }});
    if (!data.turns.length) windows.innerHTML = '<p class="empty">Timeline này không có diarization. Hãy chạy feature extraction với <code>--domain podcast</code>.</p>';
    player.addEventListener("timeupdate", () => setActive(player.currentTime));
    player.addEventListener("loadedmetadata", () => setActive(0));
  </script>
</body>
</html>"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("features_path", type=Path, help="Path to features.json")
    parser.add_argument("--video-path", type=Path, default=None, help="Defaults to ../source_video.mp4")
    parser.add_argument("--output", type=Path, default=None, help="Defaults to review.html beside features.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timeline = FeatureTimeline.model_validate_json(args.features_path.read_text(encoding="utf-8"))
    output_path = args.output or args.features_path.with_name("review.html")
    video_path = args.video_path or args.features_path.parent.parent / "source_video.mp4"
    video_source = Path(os.path.relpath(video_path, output_path.parent)).as_posix()
    output_path.write_text(build_review_html(timeline, video_source=video_source), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
