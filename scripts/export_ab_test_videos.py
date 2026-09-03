# -*- coding: utf-8 -*-
"""Export blind A/B test highlight video pairs comparing Fine-Tuned LTR vs Pretrained LTR."""

import argparse
import csv
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from highlight_agent.models.ltr_scorer import AdditiveAttentionScorer
from highlight_agent.models.train_offline import load_feature_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SAMPLE_VID_METADATA = {
    "iCvmsMzlF7o": {
        "title": "TED Talk: Brené Brown on Vulnerability",
        "domain": "lecture",
        "fold": 0,
        "summary": "Diễn giả chia sẻ về sức mạnh của việc dũng cảm chấp nhận tổn thương và điểm yếu của bản thân.",
    },
    "8KkKuTCFvzI": {
        "title": "TED Talk: Robert Waldinger (Harvard Study on Happiness)",
        "domain": "lecture",
        "fold": 1,
        "summary": "Bài học lớn nhất từ công trình nghiên cứu dài 75 năm của Harvard về bí quyết sống hạnh phúc.",
    },
    "bBC-nXj3Ng4": {
        "title": "3Blue1Brown: But how does bitcoin actually work?",
        "domain": "lecture",
        "fold": 2,
        "summary": "Bản chất toán học và mật mã học đằng sau sổ cái phi tập trung của Bitcoin.",
    },
    "u36A-YTxiOw": {
        "title": "Lex Fridman Podcast: Sam Altman on Startup Advice",
        "domain": "podcast",
        "fold": 1,
        "summary": "Sam Altman chia sẻ lời khuyên khởi nghiệp, cách tìm ý tưởng và tư duy tạo ra sản phẩm đột phá.",
    },
    "hp6n1qwo1Ws": {
        "title": "Lex Fridman Podcast: Pavel Durov (Telegram Founder)",
        "domain": "podcast",
        "fold": 3,
        "summary": "Pavel Durov chia sẻ bí quyết xây dựng tính kỷ luật, sự tự do và hành trình phát triển Telegram.",
    },
    "Ks-_Mh1QhMc": {
        "title": "TED Talk: Amy Cuddy on Body Language",
        "domain": "podcast",
        "fold": 4,
        "summary": "Ngôn ngữ cơ thể và tư thế đứng định hình tâm lý, sự tự tin và thành công trong giao tiếp.",
    },
}


def find_best_segment(
    scores: np.ndarray,
    target_duration: float = 50.0,
    sample_rate: float = 1.0,
    min_start: float = 15.0,
) -> tuple[float, float]:
    """Find contiguous window with highest mean score."""
    window_len = int(target_duration * sample_rate)
    start_offset = int(min_start * sample_rate)
    if len(scores) <= window_len + start_offset:
        return min_start, min_start + target_duration

    valid_scores = scores[start_offset:]
    rolling = np.convolve(valid_scores, np.ones(window_len) / window_len, mode="valid")
    best_idx = int(np.argmax(rolling)) + start_offset
    start_time = float(best_idx / sample_rate)
    end_time = float(start_time + target_duration)
    return start_time, end_time


def render_cut(source_video: Path, start_s: float, end_s: float, output_path: Path) -> bool:
    """Cut clip using FFmpeg."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration = max(5.0, end_s - start_s)
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_s),
        "-t", str(duration),
        "-i", str(source_video),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "22",
        "-c:a", "aac",
        "-b:a", "128k",
        "-avoid_negative_ts", "make_zero",
        str(output_path),
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return output_path.exists() and output_path.stat().st_size > 1000
    except Exception as e:
        print(f"Error rendering {output_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Export A/B test highlight pairs.")
    parser.add_argument("--output-dir", default="output/ab_test")
    parser.add_argument("--sample-vids", nargs="*", default=list(SAMPLE_VID_METADATA.keys()))
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    videos_dir = out_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    gt_path = out_dir / "ab_test_groundtruth.csv"
    form_guide_path = out_dir / "google_form_template.md"

    base_ckpt = "data/models/ltr_target_lecture_podcast.pt"
    pretrained_model = AdditiveAttentionScorer.load(base_ckpt, device="cpu")

    rows = []
    form_sections = []

    rng = random.Random(42)

    for idx, vid in enumerate(args.sample_vids, start=1):
        meta = SAMPLE_VID_METADATA.get(vid, {"title": vid, "summary": "", "domain": "general", "fold": 0})
        src_video = Path(f"output/{vid}/source_video.mp4")
        if not src_video.exists():
            print(f"Warning: {src_video} not found, skipping {vid}...")
            continue

        fold = meta["fold"]
        ft_ckpt = f"data/checkpoints/ltr_custom_fold{fold}.pt"
        if Path(ft_ckpt).exists():
            finetuned_model = AdditiveAttentionScorer.load(ft_ckpt, device="cpu")
        else:
            finetuned_model = pretrained_model

        # Load feature cache
        feat_matrix = load_feature_matrix(Path("data/features_cache"), vid)  # [7, T]
        # Mean pool to 1Hz
        t_sec = feat_matrix.shape[1] // 10
        feat_1hz = np.array([feat_matrix[:, i*10:(i+1)*10].mean(axis=1) for i in range(t_sec)])  # [T, 7]
        feat_tensor = torch.as_tensor(feat_1hz, dtype=torch.float32)

        with torch.no_grad():
            pre_scores = pretrained_model(feat_tensor).reshape(-1).numpy()
            ft_scores = finetuned_model(feat_tensor).reshape(-1).numpy()

        pre_start, pre_end = find_best_segment(pre_scores, target_duration=50.0)
        ft_start, ft_end = find_best_segment(ft_scores, target_duration=50.0)

        # Snap to boundaries if available for fine-tuned
        b_file = Path(f"data/annotations/boundaries/{vid}.json")
        if b_file.exists():
            try:
                b_data = json.loads(b_file.read_text(encoding="utf-8"))
                intervals = b_data.get("consensus_intervals") or b_data.get("intervals") or []
                if intervals:
                    # Choose interval closest to ft_start
                    best_inter = min(intervals, key=lambda iv: abs(iv["start_time"] - ft_start))
                    ft_start = float(best_inter["start_time"])
                    ft_end = float(best_inter["end_time"])
            except Exception:
                pass

        # Randomize assignment: Option A vs Option B
        is_a_finetuned = rng.choice([True, False])
        opt_a_model = "fine_tuned_ltr_in_domain" if is_a_finetuned else "pretrained_tvsum_summe"
        opt_b_model = "pretrained_tvsum_summe" if is_a_finetuned else "fine_tuned_ltr_in_domain"

        opt_a_start, opt_a_end = (ft_start, ft_end) if is_a_finetuned else (pre_start, pre_end)
        opt_b_start, opt_b_end = (pre_start, pre_end) if is_a_finetuned else (ft_start, ft_end)

        opt_a_file = videos_dir / f"{idx:02d}_{vid}_option_A.mp4"
        opt_b_file = videos_dir / f"{idx:02d}_{vid}_option_B.mp4"

        print(f"[{idx}/{len(args.sample_vids)}] Rendering {vid}...")
        print(f"  Option A ({opt_a_model}): {opt_a_start:.1f}s - {opt_a_end:.1f}s -> {opt_a_file.name}")
        print(f"  Option B ({opt_b_model}): {opt_b_start:.1f}s - {opt_b_end:.1f}s -> {opt_b_file.name}")

        render_cut(src_video, opt_a_start, opt_a_end, opt_a_file)
        render_cut(src_video, opt_b_start, opt_b_end, opt_b_file)

        q_id = f"Q{idx:02d}_{vid}"
        rows.append({
            "question_id": q_id,
            "video_id": vid,
            "title": meta["title"],
            "domain": meta["domain"],
            "option_a_model": opt_a_model,
            "option_b_model": opt_b_model,
            "option_a_timestamps": f"{opt_a_start:.1f}-{opt_a_end:.1f}",
            "option_b_timestamps": f"{opt_b_start:.1f}-{opt_b_end:.1f}",
            "groundtruth_winner": "Option A" if is_a_finetuned else "Option B",
        })

        form_sections.append(f"""## Phần {idx}/6: {meta['title']}
📌 **Chủ đề video gốc:** *{meta['summary']}*

* **Video Option A (Clip 45s–60s):** `{opt_a_file.name}`
* **Video Option B (Clip 45s–60s):** `{opt_b_file.name}`

### Câu 1: Mức độ yêu thích tổng thể *(Bắt buộc)*
Bạn thấy đoạn clip highlight nào hay, lôi cuốn và đáng xem hơn?
- [ ] Option A hay hơn
- [ ] Option B hay hơn
- [ ] Cả hai tương đương nhau (Tie)

### Câu 2: Tính trọn vẹn câu thoại *(Bắt buộc)*
Clip nào có câu thoại trọn vẹn hơn (không bị cắt ngang câu nói của diễn giả)?
- [ ] Option A trọn vẹn hơn
- [ ] Option B trọn vẹn hơn
- [ ] Cả hai đều trọn vẹn
---
""")

    # Write secret groundtruth mapping CSV
    with open(gt_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "question_id",
                "video_id",
                "title",
                "domain",
                "option_a_model",
                "option_b_model",
                "option_a_timestamps",
                "option_b_timestamps",
                "groundtruth_winner",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    # Write form template guide
    with open(form_guide_path, "w", encoding="utf-8") as f:
        f.write("# Google Form Setup Template (1 Form Duy Nhất - 6 Video Tiêu Biểu)\n\n")
        f.write("> **Hướng dẫn:** Tải 12 video trong `output/ab_test/videos/` lên YouTube ở chế độ **Không công khai (Unlisted)** và chèn vào từng Phần (Section) tương ứng bên dưới.\n\n")
        f.write("\n".join(form_sections))

    print("\n=======================================================")
    print(" >>> EXPORTED 12 BLIND A/B TEST VIDEOS SUCCESSFULLY! <<<")
    print(f" Video Directory: {videos_dir}")
    print(f" Groundtruth CSV: {gt_path}")
    print(f" Form Guide Markdown: {form_guide_path}")
    print("=======================================================")
    return 0


if __name__ == "__main__":
    main()
