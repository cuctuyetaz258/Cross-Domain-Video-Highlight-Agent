"""Tạo file CSV mẫu gán nhãn 2 giây (TVSum-style) cho video in-domain.

Tự động tạo các mốc 2 giây liên tiếp từ 0.0s đến hết video, kèm câu thoại transcript (nếu có)
để người gán nhãn chỉ cần mở file Excel / Google Sheets và điền điểm 1-5.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

# Danh sách 13 video chuẩn của dự án
CATALOG_13_VIDEOS = {
    # 5 Lectures
    "WUvTyaaNkzM": {"domain": "lecture", "duration": 1026.0, "title": "The Essence of Calculus", "pilot": True},
    "aircAruvnKk": {"domain": "lecture", "duration": 1122.0, "title": "What is a Neural Network?", "pilot": True},
    "IHZwWFHWa-w": {"domain": "lecture", "duration": 1236.0, "title": "Gradient Descent", "pilot": False},
    "wjZofJX0v4M": {"domain": "lecture", "duration": 1632.0, "title": "Transformers & LLMs", "pilot": False},
    "g2-_pnmhO4A": {"domain": "lecture", "duration": 1524.0, "title": "CS50 in 25 Minutes", "pilot": False},
    # 5 Podcasts
    "DNQDqq4mWSY": {"domain": "podcast", "duration": 690.0, "title": "Sam Altman on GPT-5", "pilot": True},
    "1bszFX_XcbU": {"domain": "podcast", "duration": 864.0, "title": "Top Study Habits - Huberman", "pilot": True},
    "waLjtcUq5Mc": {"domain": "podcast", "duration": 996.0, "title": "Tucker Carlson on Putin", "pilot": False},
    "-cRswJf8OnI": {"domain": "podcast", "duration": 1314.0, "title": "Brutal Truth About Money", "pilot": False},
    "u36A-YTxiOw": {"domain": "podcast", "duration": 1266.0, "title": "Best Way to Launch Startup", "pilot": False},
    # 3 Standups
    "88bD9f2MivI": {"domain": "standup", "duration": 900.0, "title": "Trevor Noah: Man of All Nations", "pilot": True},
    "oUUVRa2SNWU": {"domain": "standup", "duration": 954.0, "title": "Ronny Chieng Stand-Up", "pilot": True},
    "mfjnDLbCroQ": {"domain": "standup", "duration": 960.0, "title": "Bill Burr Stand-Up Comedy", "pilot": False},
}


def _get_transcript_segments(video_id: str, output_root: Path) -> list[dict[str, Any]]:
    """Đọc transcript nếu đã có sẵn trong output/<video_id>/transcript.json."""
    transcript_file = output_root / video_id / "transcript.json"
    if not transcript_file.is_file():
        return []
    try:
        data = json.loads(transcript_file.read_text(encoding="utf-8"))
        return data.get("segments", [])
    except Exception:
        return []


def _extract_hint_text(start: float, end: float, segments: list[dict[str, Any]]) -> str:
    """Trích xuất câu thoại trong khoảng [start, end]."""
    if not segments:
        return ""
    matched_texts = []
    for seg in segments:
        s_start = float(seg.get("start", 0.0))
        s_end = float(seg.get("end", 0.0))
        # Có giao thoa thời gian
        if max(start, s_start) < min(end, s_end):
            text = " ".join(seg.get("text", "").replace("\n", " ").split()).strip()
            if text and text not in matched_texts:
                matched_texts.append(text)
    return " ".join(matched_texts)[:120]  # Giới hạn 120 ký tự để dễ nhìn trong Excel


def generate_single_template(
    video_id: str,
    domain: str,
    duration: float,
    annotator_id: str,
    output_dir: Path,
    output_root: Path = Path("output"),
    step_sec: float = 2.0,
) -> Path:
    """Tạo 1 file CSV template 2 giây cho video."""
    output_dir.mkdir(parents=True, exist_ok=True)
    # The first annotation follows the canonical dataset name. Additional
    # annotators keep their identifier so multiple labels never overwrite.
    file_name = f"{video_id}.csv" if annotator_id == "annotator_1" else f"{video_id}_{annotator_id}.csv"
    out_file = output_dir / file_name

    segments = _get_transcript_segments(video_id, output_root)

    rows = []
    curr_start = 0.0
    while curr_start < duration:
        curr_end = min(curr_start + step_sec, duration)
        hint = _extract_hint_text(curr_start, curr_end, segments)

        rows.append(
            {
                "video_id": video_id,
                "start_sec": f"{curr_start:.1f}",
                "end_sec": f"{curr_end:.1f}",
                "annotator_id": annotator_id,
                "importance": "",  # Để trống cho annotator điền 1-5
                "domain": domain,
                "reason_tag": "",  # Tùy chọn: key_idea, punchline, filler...
                "transcript_hint": hint,
                "needs_review": "false",
                "label_protocol": "multimodal_tvsum_style",
            }
        )
        curr_start += step_sec

    fieldnames = [
        "video_id",
        "start_sec",
        "end_sec",
        "annotator_id",
        "importance",
        "domain",
        "reason_tag",
        "transcript_hint",
        "needs_review",
        "label_protocol",
    ]

    with open(out_file, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return out_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Tạo file CSV mẫu 2 giây gán nhãn chuẩn TVSum")
    parser.add_argument("--video-id", default=None, help="Mã ID video (nếu tạo lẻ 1 video)")
    parser.add_argument("--domain", choices=["lecture", "podcast", "standup"], default=None)
    parser.add_argument("--duration", type=float, default=None, help="Tổng thời lượng video (giây)")
    parser.add_argument(
        "--annotator-id",
        default="annotator_1",
        help="Tên hoặc mã người gán nhãn; annotator_1 tạo tên chuẩn <video_id>.csv",
    )
    parser.add_argument("--output-dir", default="data/annotations/raw", help="Thư mục xuất file CSV")
    parser.add_argument("--pilot", action="store_true", help="Tạo toàn bộ 6 video Pilot")
    parser.add_argument("--all", action="store_true", help="Tạo toàn bộ 13 video trong danh mục")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)

    targets = {}
    if args.all:
        targets = CATALOG_13_VIDEOS
    elif args.pilot:
        targets = {k: v for k, v in CATALOG_13_VIDEOS.items() if v.get("pilot")}
    elif args.video_id:
        v_info = CATALOG_13_VIDEOS.get(args.video_id, {})
        domain = args.domain or v_info.get("domain", "lecture")
        duration = args.duration or v_info.get("duration", 600.0)
        targets = {args.video_id: {"domain": domain, "duration": duration, "title": args.video_id}}
    else:
        # Mặc định tạo 6 video Pilot
        targets = {k: v for k, v in CATALOG_13_VIDEOS.items() if v.get("pilot")}

    print(f"Bắt đầu tạo {len(targets)} template gán nhãn cho annotator: '{args.annotator_id}'...")
    created_files = []
    for vid, meta in targets.items():
        file_path = generate_single_template(
            video_id=vid,
            domain=meta["domain"],
            duration=meta["duration"],
            annotator_id=args.annotator_id,
            output_dir=out_dir,
        )
        created_files.append(file_path)
        print(f"  ✓ Đã tạo: {file_path.name} ({meta['domain']} - {meta['duration'] / 60:.1f} phút)")

    print(f"\n Hoàn tất! Đã lưu {len(created_files)} file CSV vào: {out_dir.absolute()}")


if __name__ == "__main__":
    main()
