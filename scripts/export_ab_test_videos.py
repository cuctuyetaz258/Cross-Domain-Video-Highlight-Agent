# -*- coding: utf-8 -*-
"""Export blind A/B test highlight pairs for human evaluation."""

import argparse
import csv
import json
import os
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Prepare blind A/B test metadata.")
    parser.add_argument("--output-dir", default="output/ab_test")
    parser.add_argument("--sample-vids", nargs="*", default=[])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    gt_path = os.path.join(args.output_dir, "ab_test_groundtruth.csv")
    form_guide_path = os.path.join(args.output_dir, "google_form_template.md")

    # Load selected videos if not provided
    vids = args.sample_vids
    if not vids:
        sel_path = "data/qvhighlights/selected_vids.json"
        if os.path.exists(sel_path):
            with open(sel_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                vids = list(data.keys())[:10]  # Take 10 sample videos for testing
        else:
            vids = ["sample_video_01", "sample_video_02", "sample_video_03"]

    rows = []
    form_questions = []

    rng = random.Random(42)

    for idx, vid in enumerate(vids, start=1):
        # Randomize whether Option A is Pretrained or Fine-tuned
        is_a_finetuned = rng.choice([True, False])
        opt_a = "fine_tuned_qvhighlights" if is_a_finetuned else "pretrained_tvsum_summe"
        opt_b = "pretrained_tvsum_summe" if is_a_finetuned else "fine_tuned_qvhighlights"

        q_id = f"Q{idx:02d}_{vid}"
        rows.append({
            "question_id": q_id,
            "video_id": vid,
            "option_a_model": opt_a,
            "option_b_model": opt_b,
            "groundtruth_winner_candidate": "A" if is_a_finetuned else "B"
        })

        form_questions.append(f"""### Câu hỏi {idx}: Video `{vid}`
- **Video Option A**: `output/ab_test/{vid}_option_A.mp4`
- **Video Option B**: `output/ab_test/{vid}_option_B.mp4`
- **Lựa chọn khảo sát (Multiple choice):**
  - [ ] Option A tốt hơn (Highlight tự nhiên và bao quát hơn)
  - [ ] Option B tốt hơn (Highlight tự nhiên và bao quát hơn)
  - [ ] Cả 2 đều ngang nhau (Hòa)
""")

    # Write groundtruth CSV
    with open(gt_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["question_id", "video_id", "option_a_model", "option_b_model", "groundtruth_winner_candidate"])
        writer.writeheader()
        writer.writerows(rows)

    # Write form guide
    with open(form_guide_path, "w", encoding="utf-8") as f:
        f.write("# Hướng dẫn tạo Google Form cho Blind A/B Testing\n\n")
        f.write("Tải các video cắt highlight (Option A và Option B) lên Google Drive và tạo Form khảo sát theo các câu hỏi dưới đây:\n\n")
        f.write("\n".join(form_questions))

    print(f"Generated ground truth secret file: {gt_path}")
    print(f"Generated Google Form question template: {form_guide_path}")
    return 0


if __name__ == "__main__":
    main()
