"""Script dùng để tạo file candidates.json từ transcript"""

import argparse
import json
from pathlib import Path

# Nạp file môi trường (.env) và hàm lõi AI
from dotenv import load_dotenv

from highlight_agent.backend import load_transcript
from highlight_agent.llm.extractor import extract_highlights_from_transcript


def parse_args():
    parser = argparse.ArgumentParser(description="Chạy AI tạo file candidates.json")
    parser.add_argument("transcript_path", help="Đường dẫn tới file transcript.json (VD: output/ABC/transcript.json)")
    parser.add_argument("--output", default="candidates.json", help="Tên file kết quả lưu ra")
    parser.add_argument(
        "--domain",
        choices=["auto", "lecture", "podcast", "standup"],
        default="auto",
        help="Miền nội dung của video (auto, lecture, podcast, standup)",
    )
    parser.add_argument("--count", type=int, default=3, help="Số lượng highlight cần trích xuất (mặc định: 3)")
    return parser.parse_args()


def format_compact_transcript(transcript_obj) -> str:
    """Gom các phân đoạn câu thoại ngắn thành các khối tự nhiên để tiết kiệm ~40% tokens, giúp LLM đọc được video dài gấp đôi."""
    lines = []
    curr_start = None
    curr_end = None
    curr_texts = []
    for s in transcript_obj.segments:
        if curr_start is None:
            curr_start = s.start
        curr_end = s.end
        txt = s.text.strip()
        if txt:
            curr_texts.append(txt)
        if (curr_end - curr_start >= 20.0) or (
            curr_texts and curr_texts[-1].endswith((".", "?", "!")) and curr_end - curr_start >= 10.0
        ):
            block_text = " ".join(curr_texts)
            lines.append(f"[{curr_start:.1f}s - {curr_end:.1f}s]: {block_text}")
            curr_start = None
            curr_texts = []
    if curr_texts:
        block_text = " ".join(curr_texts)
        lines.append(f"[{curr_start:.1f}s - {curr_end:.1f}s]: {block_text}")
    return "\n".join(lines)


def main():
    load_dotenv()
    args = parse_args()

    # 1. Load transcript
    print(f"Đang đọc file transcript: {args.transcript_path}")
    transcript_obj = load_transcript(args.transcript_path)

    # Gom các câu thoại theo định dạng gọn gàng
    full_text = format_compact_transcript(transcript_obj)

    # 2. Gọi hàm lõi AI của bạn
    print(
        f"Đang gửi cho AI phân tích ({args.count} highlights, domain: {args.domain}, độ dài transcript: {len(full_text)} ký tự)..."
    )
    candidates = extract_highlights_from_transcript(full_text, domain=args.domain, highlight_count=args.count)

    # 3. Lưu kết quả ra thành file .json vật lý
    out_path = Path(args.output)
    # Chuyển đổi dữ liệu về dạng từ điển để lưu JSON
    json_list = [c.model_dump(mode="json") for c in candidates]

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(json_list, f, ensure_ascii=False, indent=2)

    print(f"Thành công! Đã tạo ra file vật lý: {out_path.absolute()}")


if __name__ == "__main__":
    main()
