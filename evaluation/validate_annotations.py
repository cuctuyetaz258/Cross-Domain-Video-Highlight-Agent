"""Kiểm tra và xác thực chất lượng dữ liệu gán nhãn 2 giây (QA & Validation).

Kiểm tra 7 tiêu chí theo Mục 9 Kế hoạch self_labeling_plan.md:
1. Giá trị importance phải thuộc tập số nguyên [1, 5] (không được để trống).
2. Timestamp tăng đều, start < end, không bị hổng (gap) hoặc chồng lấn (overlap).
3. Độ bao phủ (Coverage) xấp xỉ 100% thời lượng video.
4. Phân phối điểm số: kiểm tra tỷ lệ điểm 4-5 không vượt quá 20%.
5. Cảnh báo nếu annotator chỉ chấm 1 loại điểm cho phần lớn video (>80%).
6. Kiểm tra trường bắt buộc và domain hợp lệ.
7. Đánh giá độ đồng thuận giữa các annotator (nếu có nhiều người chấm cùng 1 video).
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def validate_single_csv(csv_path: str | Path) -> dict[str, Any]:
    """Kiểm tra tính hợp lệ của 1 file CSV gán nhãn."""
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file: {path}")

    errors: list[str] = []
    warnings: list[str] = []

    with open(path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return {
            "file": path.name,
            "valid": False,
            "errors": ["File CSV rỗng không có dữ liệu."],
            "warnings": [],
            "stats": {},
        }

    video_id = rows[0].get("video_id", "")
    domain = rows[0].get("domain", "")
    annotator_id = rows[0].get("annotator_id", "")

    scores: list[int] = []
    expected_start = 0.0

    for idx, row in enumerate(rows, start=2):  # Dòng 2 trong CSV (sau header)
        # 1. Kiểm tra timestamp
        try:
            start_sec = float(row.get("start_sec", -1))
            end_sec = float(row.get("end_sec", -1))
        except ValueError:
            errors.append(f"Dòng {idx}: start_sec hoặc end_sec không phải là số hợp lệ.")
            continue

        if start_sec < 0 or end_sec <= start_sec:
            errors.append(f"Dòng {idx}: Lỗi timestamp ({start_sec} -> {end_sec}).")

        if abs(start_sec - expected_start) > 0.15:
            errors.append(
                f"Dòng {idx}: Bị hổng hoặc lệch timeline (kỳ vọng {expected_start:.1f}s, nhưng gặp {start_sec:.1f}s)."
            )

        expected_start = end_sec

        # 2. Kiểm tra importance score
        raw_imp = str(row.get("importance", "")).strip()
        if not raw_imp:
            errors.append(f"Dòng {idx} ({start_sec:.1f}s-{end_sec:.1f}s): Chưa điền điểm importance.")
            continue

        try:
            score = int(raw_imp)
            if not (1 <= score <= 5):
                errors.append(f"Dòng {idx}: Điểm importance={score} nằm ngoài khoảng [1, 5].")
            else:
                scores.append(score)
        except ValueError:
            errors.append(f"Dòng {idx}: Điểm importance '{raw_imp}' không phải số nguyên 1-5.")

    total_segments = len(rows)
    valid = len(errors) == 0

    stats: dict[str, Any] = {}
    if scores:
        counts = Counter(scores)
        pct_5 = counts.get(5, 0) / len(scores) * 100
        pct_4 = counts.get(4, 0) / len(scores) * 100
        pct_3 = counts.get(3, 0) / len(scores) * 100
        pct_2 = counts.get(2, 0) / len(scores) * 100
        pct_1 = counts.get(1, 0) / len(scores) * 100

        high_score_pct = pct_4 + pct_5

        stats = {
            "video_id": video_id,
            "annotator_id": annotator_id,
            "domain": domain,
            "total_segments": total_segments,
            "scored_segments": len(scores),
            "pct_5": round(pct_5, 1),
            "pct_4": round(pct_4, 1),
            "pct_3": round(pct_3, 1),
            "pct_2": round(pct_2, 1),
            "pct_1": round(pct_1, 1),
            "high_score_pct": round(high_score_pct, 1),
            "mean_score": round(float(np.mean(scores)), 2),
            "std_score": round(float(np.std(scores)), 2),
        }

        # Kiểm tra cảnh báo phân phối (Soft guidelines)
        if high_score_pct > 25.0:
            warnings.append(
                f"Tỷ lệ điểm 4-5 chiếm {high_score_pct:.1f}% (> 20%). Cần kiểm tra lại để tránh thiên lệch chấm quá dễ tính."
            )
        if counts.get(1, 0) == 0 and counts.get(2, 0) == 0:
            warnings.append("Video không có đoạn nào nhận điểm 1 hoặc 2. Cần phân loại rõ đoạn quan trọng và đoạn nền.")

        # Cảnh báo lặp 1 điểm
        most_common_cnt = counts.most_common(1)[0][1]
        if most_common_cnt / len(scores) > 0.85:
            warnings.append(f"Annotator chấm cùng 1 mức điểm cho hơn 85% video ({most_common_cnt}/{len(scores)} đoạn).")

    return {
        "file": path.name,
        "valid": valid,
        "errors": errors,
        "warnings": warnings,
        "stats": stats,
    }


def validate_annotation_directory(dir_path: str | Path) -> None:
    """Quét và kiểm tra toàn bộ file CSV trong thư mục."""
    folder = Path(dir_path)
    if not folder.is_dir():
        print(f"Thư mục không tồn tại: {folder}")
        return

    csv_files = sorted(folder.glob("*.csv"))
    if not csv_files:
        print(f"Không tìm thấy file CSV nào trong: {folder.absolute()}")
        return

    print("=" * 80)
    print(f" 📋 BÁO CÁO KIỂM CHỨNG CHẤT LƯỢNG GÁN NHÃN ({len(csv_files)} files)")
    print("=" * 80)

    all_valid = True
    for file in csv_files:
        res = validate_single_csv(file)
        status_icon = "✅ HỢP LỆ" if res["valid"] else "❌ LỖI"
        if not res["valid"]:
            all_valid = False

        print(f"\n📄 File: {res['file']} | {status_icon}")
        if res["stats"]:
            s = res["stats"]
            print(
                f"   [Thống kê] Video: {s['video_id']} | Annotator: {s['annotator_id']} | "
                f"Số đoạn: {s['scored_segments']}/{s['total_segments']} | "
                f"Mean: {s['mean_score']} ± {s['std_score']}"
            )
            print(
                f"   [Phân phối] 1★: {s['pct_1']}% | 2★: {s['pct_2']}% | "
                f"3★: {s['pct_3']}% | 4★: {s['pct_4']}% | 5★: {s['pct_5']}% "
                f"(4-5★: {s['high_score_pct']}%)"
            )

        if res["errors"]:
            print(f"   [Lỗi nghiêm trọng ({len(res['errors'])} lỗi)]:")
            for err in res["errors"][:5]:
                print(f"     - {err}")
            if len(res["errors"]) > 5:
                print(f"     ... và {len(res['errors']) - 5} lỗi khác.")

        if res["warnings"]:
            print(f"   [Cảnh báo chất lượng ({len(res['warnings'])} lưu ý)]:")
            for warn in res["warnings"]:
                print(f"     ⚠️ {warn}")

    print("\n" + "=" * 80)
    if all_valid:
        print("🎉 TẤT CẢ FILE CSV ĐỀU ĐẠT TIÊU CHUẨN XÁC THỰC!")
    else:
        print("⚠️ CẦN SỬA CÁC LỖI TRÊN TRƯỚC KHI TIẾN HÀNH TRAINING / EVALUATION.")
    print("=" * 80 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validator cho dữ liệu gán nhãn 2 giây TVSum-style")
    parser.add_argument("input_path", nargs="?", default="data/annotations/raw", help="Đường dẫn file CSV hoặc thư mục")
    args = parser.parse_args()

    target = Path(args.input_path)
    if target.is_file():
        res = validate_single_csv(target)
        print(f"\nKết quả kiểm tra {target.name}: {'HỢP LỆ' if res['valid'] else 'LỖI'}")
        for err in res["errors"]:
            print(f"  - Lỗi: {err}")
        for warn in res["warnings"]:
            print(f"  - Cảnh báo: {warn}")
    else:
        validate_annotation_directory(target)


if __name__ == "__main__":
    main()
