# -*- coding: utf-8 -*-
"""Calculate Human Preference Win-Rate from Google Form survey responses."""

import argparse
import csv
import json
import os
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(description="Calculate Human Preference Win-Rate from Google Form.")
    parser.add_argument("--responses-csv", required=True, help="Path to exported CSV from Google Forms.")
    parser.add_argument("--groundtruth-csv", default="output/ab_test/ab_test_groundtruth.csv", help="Secret mapping CSV.")
    parser.add_argument("--output-report", default="data/reports/human_preference_report.json")
    args = parser.parse_args()

    if not os.path.exists(args.groundtruth_csv):
        print(f"Error: {args.groundtruth_csv} not found.")
        return 1

    gt_map = {}
    with open(args.groundtruth_csv, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            gt_map[row["question_id"]] = row

    if not os.path.exists(args.responses_csv):
        print(f"Error: {args.responses_csv} not found.")
        return 1

    with open(args.responses_csv, "r", encoding="utf-8-sig") as f:
        responses = list(csv.DictReader(f))

    print(f"Loaded {len(responses)} survey responses.")

    total_votes = 0
    finetuned_wins = 0
    pretrained_wins = 0
    ties = 0

    per_video_stats = defaultdict(lambda: {"finetuned": 0, "pretrained": 0, "tie": 0, "total": 0})

    for resp in responses:
        for col_name, val in resp.items():
            val_clean = str(val).strip().lower()
            if not val_clean:
                continue

            # Match question
            matched_qid = None
            for qid in gt_map:
                if qid in col_name or gt_map[qid]["video_id"] in col_name:
                    matched_qid = qid
                    break

            if not matched_qid:
                continue

            gt = gt_map[matched_qid]
            vid = gt["video_id"]

            total_votes += 1
            per_video_stats[vid]["total"] += 1

            if "hòa" in val_clean or "ngang nhau" in val_clean or "tie" in val_clean:
                ties += 1
                per_video_stats[vid]["tie"] += 1
            elif "option a" in val_clean or val_clean.startswith("a"):
                if gt["option_a_model"] == "fine_tuned_qvhighlights":
                    finetuned_wins += 1
                    per_video_stats[vid]["finetuned"] += 1
                else:
                    pretrained_wins += 1
                    per_video_stats[vid]["pretrained"] += 1
            elif "option b" in val_clean or val_clean.startswith("b"):
                if gt["option_b_model"] == "fine_tuned_qvhighlights":
                    finetuned_wins += 1
                    per_video_stats[vid]["finetuned"] += 1
                else:
                    pretrained_wins += 1
                    per_video_stats[vid]["pretrained"] += 1

    finetuned_winrate = (finetuned_wins / total_votes * 100) if total_votes > 0 else 0.0
    pretrained_winrate = (pretrained_wins / total_votes * 100) if total_votes > 0 else 0.0
    tie_rate = (ties / total_votes * 100) if total_votes > 0 else 0.0

    report = {
        "total_respondents": len(responses),
        "total_evaluated_votes": total_votes,
        "finetuned_qvhighlights": {
            "wins": finetuned_wins,
            "win_rate_percent": round(finetuned_winrate, 2)
        },
        "pretrained_tvsum_summe": {
            "wins": pretrained_wins,
            "win_rate_percent": round(pretrained_winrate, 2)
        },
        "tie": {
            "count": ties,
            "rate_percent": round(tie_rate, 2)
        },
        "per_video_breakdown": per_video_stats
    }

    os.makedirs(os.path.dirname(args.output_report), exist_ok=True)
    with open(args.output_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n========== KẾT QUẢ KHẢO SÁT HUMAN PREFERENCE (USER METRIC) ==========")
    print(f"Tổng số phiếu đánh giá: {total_votes}")
    print(f"Tỉ lệ chọn Fine-tuned (QVHighlights): {finetuned_winrate:.2f}% ({finetuned_wins} phiếu)")
    print(f"Tỉ lệ chọn Pretrained (TVSum+SumMe):   {pretrained_winrate:.2f}% ({pretrained_wins} phiếu)")
    print(f"Tỉ lệ Hòa (Ngang nhau):                {tie_rate:.2f}% ({ties} phiếu)")
    print(f"Báo cáo chi tiết đã lưu tại: {args.output_report}")
    print("====================================================================\n")
    return 0


if __name__ == "__main__":
    main()
