"""Đánh giá chất lượng Highlight trên tập dữ liệu In-Domain (10 video thực tế của nhóm).

Hỗ trợ so sánh các phương pháp:
- 'llm_baseline'   : Trích xuất từ LLM (extract_highlights.py / candidates.json)
- 'random'         : Chọn ngẫu nhiên 3 đoạn thời lượng 45s
- 'uniform'        : Chia đều 3 đoạn cách quãng trên toàn bộ video
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.metrics import (
    compute_hit_at_k,
    compute_mean_iou,
    compute_temporal_precision_recall_f1,
    temporal_iou,
)


def load_ground_truth(gt_path: str | Path) -> dict[str, Any]:
    """Đọc file ground truth (CSV 2 giây TVSum-style hoặc JSON) của 1 video."""
    path = Path(gt_path)
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file Ground Truth: {path}")

    if path.suffix.lower() == ".csv":
        import csv
        with open(path, encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            raise ValueError(f"File CSV rỗng: {path}")

        video_id = rows[0].get("video_id", path.stem.split("_")[0])
        domain = rows[0].get("domain", "unknown")
        annotator = rows[0].get("annotator_id", "anonymous")
        duration = float(rows[-1].get("end_sec", 0.0))

        # Gom các đoạn có importance >= 4 liên tiếp thành highlight intervals
        highlights: list[dict[str, Any]] = []
        curr_hl: list[tuple[float, float, int]] = []

        for r in rows:
            imp_str = str(r.get("importance", "")).strip()
            imp = int(imp_str) if imp_str.isdigit() else 0
            if imp >= 4:
                curr_hl.append((float(r["start_sec"]), float(r["end_sec"]), imp))
            else:
                if curr_hl:
                    st = curr_hl[0][0]
                    en = curr_hl[-1][1]
                    max_imp = max(x[2] for x in curr_hl)
                    highlights.append({
                        "start_time": st,
                        "end_time": en,
                        "importance_score": max_imp,
                    })
                    curr_hl = []

        if curr_hl:
            st = curr_hl[0][0]
            en = curr_hl[-1][1]
            max_imp = max(x[2] for x in curr_hl)
            highlights.append({
                "start_time": st,
                "end_time": en,
                "importance_score": max_imp,
            })

        return {
            "video_id": video_id,
            "title": video_id,
            "domain": domain,
            "annotator": annotator,
            "duration": duration,
            "highlights": highlights,
        }

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    # Chuẩn hóa danh sách highlights
    highlights = data.get("highlights", [])
    for h in highlights:
        h["start_time"] = float(h.get("start_time", 0.0))
        h["end_time"] = float(h.get("end_time", 0.0))

    return {
        "video_id": data.get("video_id", path.stem),
        "title": data.get("title", path.stem),
        "domain": data.get("domain", "unknown"),
        "annotator": data.get("annotator", "anonymous"),
        "duration": float(data.get("duration", 0.0)),
        "highlights": highlights,
    }


def load_all_ground_truths(gt_dir: str | Path) -> dict[str, dict[str, Any]]:
    """Đọc toàn bộ file Ground Truth (CSV hoặc JSON) trong thư mục."""
    directory = Path(gt_dir)
    if not directory.is_dir():
        raise NotADirectoryError(f"Thư mục Ground Truth không tồn tại: {directory}")

    gt_dict: dict[str, dict[str, Any]] = {}
    files = sorted(list(directory.glob("*.csv")) + list(directory.glob("*.json")))
    for file in files:
        try:
            gt_data = load_ground_truth(file)
            if gt_data["highlights"]:  # Chỉ lấy video đã hoàn tất gán nhãn
                gt_dict[gt_data["video_id"]] = gt_data
        except Exception as e:
            print(f"[CẢNH BÁO] Không thể đọc file Ground Truth '{file.name}': {e}")

    return gt_dict



def generate_baseline_predictions(
    video_id: str,
    duration: float,
    method: str = "random",
    count: int = 3,
    clip_duration: float = 45.0,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Tạo dự đoán giả lập cho các Baseline so sánh (Random / Uniform)."""
    if duration <= 0:
        duration = 600.0  # Fallback 10 phút nếu thiếu metadata duration

    predictions: list[dict[str, Any]] = []

    if method == "uniform":
        # Chia đều video thành count phần
        step = (duration - clip_duration) / (count + 1) if duration > clip_duration else 0.0
        for i in range(1, count + 1):
            start = round(float(i * step), 1)
            end = round(start + clip_duration, 1)
            predictions.append({
                "candidate_id": f"unif_{i:02d}",
                "start_time": start,
                "end_time": min(end, duration),
                "score": 5.0,
            })
    elif method == "random":
        rng = np.random.default_rng(seed + sum(ord(c) for c in video_id))
        max_start = max(1.0, duration - clip_duration)
        random_starts = sorted(rng.uniform(0.0, max_start, size=count * 2))

        chosen_starts: list[float] = []
        for s in random_starts:
            if not any(abs(s - c) < clip_duration for c in chosen_starts):
                chosen_starts.append(s)
            if len(chosen_starts) == count:
                break

        while len(chosen_starts) < count:
            chosen_starts.append(float(rng.uniform(0.0, max_start)))

        for i, start in enumerate(sorted(chosen_starts), start=1):
            end = round(start + clip_duration, 1)
            predictions.append({
                "candidate_id": f"rand_{i:02d}",
                "start_time": round(start, 1),
                "end_time": min(end, duration),
                "score": 5.0,
            })
    else:
        raise ValueError(f"Baseline method không hỗ trợ: {method}")

    return predictions


def load_predictions(
    video_id: str,
    pred_dir: str | Path,
    method: str = "llm_baseline",
    duration: float = 600.0,
    count: int = 3,
) -> list[dict[str, Any]]:
    """Đọc dự đoán từ file candidates.json hoặc sinh baseline giả lập."""
    if method in {"random", "uniform"}:
        return generate_baseline_predictions(video_id, duration, method=method, count=count)

    pred_path = Path(pred_dir)
    # Tìm kiếm theo nhiều đường dẫn khả dĩ:
    # 1. pred_dir / <video_id> / candidates.json
    # 2. pred_dir / <video_id>.json
    # 3. pred_dir / <video_id> / output / candidates.json
    candidates_file = pred_path / video_id / "candidates.json"
    if not candidates_file.is_file():
        candidates_file = pred_path / f"{video_id}.json"

    if not candidates_file.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy file dự đoán cho video '{video_id}' tại '{pred_path / video_id / 'candidates.json'}'"
        )

    with open(candidates_file, encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict) and "highlights" in raw:
        items = raw["highlights"]
    elif isinstance(raw, list):
        items = raw
    else:
        raise ValueError(f"Định dạng file dự đoán không hợp lệ: {candidates_file}")

    parsed = []
    for item in items:
        parsed.append({
            "candidate_id": item.get("candidate_id", "hl"),
            "start_time": float(item.get("start_time", 0.0)),
            "end_time": float(item.get("end_time", 0.0)),
            "score": float(item.get("score", 0.0)),
            "reason": item.get("reason", ""),
        })

    # Sắp xếp theo score giảm dần
    parsed.sort(key=lambda x: x["score"], reverse=True)
    return parsed


def evaluate_single_video(
    predictions: list[dict[str, Any]],
    ground_truths: list[dict[str, Any]],
    k: int = 3,
) -> dict[str, float]:
    """Đánh giá chi tiết cho 1 video với các chỉ số Hit@K, F1@K, Mean IoU."""
    hit1_03 = compute_hit_at_k(predictions, ground_truths, k=1, iou_threshold=0.3)
    hit3_03 = compute_hit_at_k(predictions, ground_truths, k=3, iou_threshold=0.3)
    hit3_05 = compute_hit_at_k(predictions, ground_truths, k=3, iou_threshold=0.5)
    hit5_03 = compute_hit_at_k(predictions, ground_truths, k=min(5, len(predictions)), iou_threshold=0.3)

    prec3, rec3, f1_3 = compute_temporal_precision_recall_f1(
        predictions, ground_truths, k=k, iou_threshold=0.3
    )
    mean_iou = compute_mean_iou(predictions, ground_truths, k=k)

    return {
        "hit@1_iou0.3": float(hit1_03),
        "hit@3_iou0.3": float(hit3_03),
        "hit@3_iou0.5": float(hit3_05),
        "hit@5_iou0.3": float(hit5_03),
        "precision@3": float(prec3),
        "recall@3": float(rec3),
        "f1@3": float(f1_3),
        "mean_iou": float(mean_iou),
    }


def evaluate_indomain(
    gt_dir: str | Path,
    pred_dir: str | Path = "output",
    method: str = "llm_baseline",
    k: int = 3,
) -> dict[str, Any]:
    """Chạy đánh giá toàn bộ tập dữ liệu In-Domain."""
    all_gts = load_all_ground_truths(gt_dir)
    if not all_gts:
        raise RuntimeError(f"Không tìm thấy file Ground Truth nào trong: {gt_dir}")

    per_video_results: list[dict[str, Any]] = []
    missing_videos: list[str] = []

    for video_id, gt_data in all_gts.items():
        domain = gt_data["domain"]
        duration = gt_data["duration"]
        gts = gt_data["highlights"]

        try:
            preds = load_predictions(video_id, pred_dir, method=method, duration=duration, count=k)
            metrics = evaluate_single_video(preds, gts, k=k)
            per_video_results.append({
                "video_id": video_id,
                "title": gt_data["title"],
                "domain": domain,
                "num_gt": len(gts),
                "num_pred": len(preds),
                **metrics,
            })
        except FileNotFoundError:
            missing_videos.append(video_id)

    if not per_video_results:
        raise RuntimeError(
            f"Không tìm thấy dự đoán nào cho các video trong Ground Truth (thiếu: {missing_videos})"
        )

    # Tính trung bình theo từng Domain
    domains = sorted(list({r["domain"] for r in per_video_results}))
    domain_summaries: dict[str, dict[str, float]] = {}

    for dom in domains:
        dom_records = [r for r in per_video_results if r["domain"] == dom]
        domain_summaries[dom] = {
            "num_videos": len(dom_records),
            "hit@1_iou0.3": float(np.mean([r["hit@1_iou0.3"] for r in dom_records])),
            "hit@3_iou0.3": float(np.mean([r["hit@3_iou0.3"] for r in dom_records])),
            "hit@3_iou0.5": float(np.mean([r["hit@3_iou0.5"] for r in dom_records])),
            "precision@3": float(np.mean([r["precision@3"] for r in dom_records])),
            "recall@3": float(np.mean([r["recall@3"] for r in dom_records])),
            "f1@3": float(np.mean([r["f1@3"] for r in dom_records])),
            "mean_iou": float(np.mean([r["mean_iou"] for r in dom_records])),
        }

    # Tổng thể (Overall)
    overall_summary = {
        "num_videos": len(per_video_results),
        "hit@1_iou0.3": float(np.mean([r["hit@1_iou0.3"] for r in per_video_results])),
        "hit@3_iou0.3": float(np.mean([r["hit@3_iou0.3"] for r in per_video_results])),
        "hit@3_iou0.5": float(np.mean([r["hit@3_iou0.5"] for r in per_video_results])),
        "precision@3": float(np.mean([r["precision@3"] for r in per_video_results])),
        "recall@3": float(np.mean([r["recall@3"] for r in per_video_results])),
        "f1@3": float(np.mean([r["f1@3"] for r in per_video_results])),
        "mean_iou": float(np.mean([r["mean_iou"] for r in per_video_results])),
    }

    return {
        "method": method,
        "evaluated_videos": len(per_video_results),
        "missing_videos": missing_videos,
        "results_per_video": per_video_results,
        "domain_summaries": domain_summaries,
        "overall_summary": overall_summary,
    }


def print_evaluation_report(results: dict[str, Any]) -> None:
    """In bảng báo cáo kết quả đánh giá định dạng Markdown chuẩn."""
    method = results["method"]
    print("\n" + "=" * 85)
    print(f" 📊 KẾT QUẢ ĐÁNH GIÁ IN-DOMAIN DATASET | PHƯƠNG PHÁP: {method.upper()}")
    print("=" * 85)

    if results["missing_videos"]:
        print(f"⚠️ Chưa có kết quả cho {len(results['missing_videos'])} video: {', '.join(results['missing_videos'])}")

    # Bảng chi tiết từng video
    print("\n### 1. Chi tiết từng Video:")
    print(f"{'Video ID':<13} | {'Domain':<8} | {'Hit@1 (0.3)':<11} | {'Hit@3 (0.3)':<11} | {'Hit@3 (0.5)':<11} | {'F1@3':<8} | {'Mean IoU':<8}")
    print("-" * 85)
    for r in results["results_per_video"]:
        print(
            f"{r['video_id']:<13} | {r['domain']:<8} | "
            f"{r['hit@1_iou0.3'] * 100:>9.1f}% | "
            f"{r['hit@3_iou0.3'] * 100:>9.1f}% | "
            f"{r['hit@3_iou0.5'] * 100:>9.1f}% | "
            f"{r['f1@3'] * 100:>6.1f}% | "
            f"{r['mean_iou']:>8.3f}"
        )

    # Bảng tổng hợp theo Domain & Overall
    print("\n" + "=" * 85)
    print("### 2. BẢNG TỔNG HỢP BÁO CÁO (THEO MIỀN & TOÀN BỘ DATASET):")
    print("=" * 85)
    print(f"{'Domain / Scope':<16} | {'Videos':<6} | {'Hit@1 (0.3)':<11} | {'Hit@3 (0.3)':<11} | {'Hit@3 (0.5)':<11} | {'F1-Score':<9} | {'Mean IoU':<8}")
    print("-" * 85)

    for dom, summary in results["domain_summaries"].items():
        print(
            f"{dom.capitalize():<16} | {summary['num_videos']:<6} | "
            f"{summary['hit@1_iou0.3'] * 100:>9.1f}% | "
            f"{summary['hit@3_iou0.3'] * 100:>9.1f}% | "
            f"{summary['hit@3_iou0.5'] * 100:>9.1f}% | "
            f"{summary['f1@3'] * 100:>7.1f}% | "
            f"{summary['mean_iou']:>8.3f}"
        )

    overall = results["overall_summary"]
    print("-" * 85)
    print(
        f"{'⭐ OVERALL ALL':<16} | {overall['num_videos']:<6} | "
        f"{overall['hit@1_iou0.3'] * 100:>9.1f}% | "
        f"{overall['hit@3_iou0.3'] * 100:>9.1f}% | "
        f"{overall['hit@3_iou0.5'] * 100:>9.1f}% | "
        f"{overall['f1@3'] * 100:>7.1f}% | "
        f"{overall['mean_iou']:>8.3f}"
    )
    print("=" * 85 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="In-Domain Dataset Evaluation Pipeline")
    parser.add_argument(
        "--gt-dir",
        default="data/annotations/raw",
        help="Thư mục chứa các file nhãn chuẩn Ground Truth (.csv hoặc .json)",
    )
    parser.add_argument(
        "--pred-dir",
        default="output",
        help="Thư mục chứa kết quả dự đoán (VD: output/ hoặc thư mục candidates)",
    )
    parser.add_argument(
        "--method",
        choices=["llm_baseline", "random", "uniform"],
        default="llm_baseline",
        help="Phương pháp cần đánh giá (llm_baseline, random, uniform)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Số lượng highlight đánh giá (Top K, mặc định: 3)",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Đường dẫn file lưu kết quả JSON chi tiết (nếu cần)",
    )
    args = parser.parse_args()

    results = evaluate_indomain(
        gt_dir=args.gt_dir,
        pred_dir=args.pred_dir,
        method=args.method,
        k=args.k,
    )

    print_evaluation_report(results)

    if args.output_json:
        out_file = Path(args.output_json)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Đã lưu kết quả chi tiết ra: {out_file.absolute()}")


if __name__ == "__main__":
    main()
