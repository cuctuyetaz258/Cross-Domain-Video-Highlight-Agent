"""Benchmark evaluation script for TVSum and SumMe according to the proposal specs."""

import argparse
from pathlib import Path

import h5py
import numpy as np

from evaluation.metrics import compute_correlation, compute_fscore, generate_summary_from_scores


def evaluate_single_dataset(
    h5_path: Path,
    dataset_name: str,
    method: str = "visual_variance",
    fscore_mode: str = "mean",
) -> dict:
    """
    Evaluates a single .h5 benchmark dataset.
    Methods:
        - 'random': Uniform random baseline
        - 'visual_variance': Feature L2 norm (visual activity/saliency baseline)
        - 'uniform': Constant baseline
    """
    if not h5_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {h5_path}")

    kendall_list = []
    spearman_list = []
    f1_list = []
    prec_list = []
    rec_list = []

    with h5py.File(h5_path, "r") as f:
        video_keys = sorted(list(f.keys()))
        print(f"\n--- Evaluating {dataset_name} ({len(video_keys)} videos) | Method: {method} ---")

        for idx, key in enumerate(video_keys):
            video_data = f[key]

            features = np.array(video_data["features"])  # (N, 1024)
            gtscore = np.array(video_data["gtscore"])  # (N,)
            user_summary = np.array(video_data["user_summary"])  # (U, N)
            change_points = np.array(video_data["change_points"])
            n_frames = int(video_data["n_frames"][...])
            n_steps = int(video_data["n_steps"][...])
            picks = np.array(video_data["picks"])

            n_samples = len(features)

            # 1. Generate predicted scores
            if method == "random":
                np.random.seed(42 + idx)
                pred_scores = np.random.rand(n_samples)
            elif method == "visual_variance":
                norms = np.linalg.norm(features, axis=1)
                pred_scores = (norms - norms.min()) / (norms.max() - norms.min() + 1e-8)
            elif method == "uniform":
                pred_scores = np.ones(n_samples)
            else:
                raise ValueError(f"Unsupported method: {method}")

            # 2. Compute Ranking Correlation (Kendall's tau & Spearman's rho)
            tau, _ = compute_correlation(pred_scores, gtscore, method="kendall")
            rho, _ = compute_correlation(pred_scores, gtscore, method="spearman")
            kendall_list.append(tau)
            spearman_list.append(rho)

            # 3. Generate 15% Summary via Knapsack and Compute F-Score
            pred_summary = generate_summary_from_scores(
                pred_scores, change_points, n_frames, n_steps, picks, budget_ratio=0.15
            )
            f1, p, r = compute_fscore(pred_summary, user_summary, eval_metric=fscore_mode)
            f1_list.append(f1)
            prec_list.append(p)
            rec_list.append(r)

    results = {
        "dataset": dataset_name,
        "method": method,
        "num_videos": len(video_keys),
        "kendall_tau": float(np.mean(kendall_list)),
        "spearman_rho": float(np.mean(spearman_list)),
        "f1_score": float(np.mean(f1_list)),
        "precision": float(np.mean(prec_list)),
        "recall": float(np.mean(rec_list)),
    }

    print(f"\n[SUMMARY RESULT -- {dataset_name}]")
    print(f" * Kendall's tau (tau)  : {results['kendall_tau']:.4f}")
    print(f" * Spearman's rho (rho) : {results['spearman_rho']:.4f}")
    print(
        f" * F1-Score (overlap)   : {results['f1_score'] * 100:.2f}% "
        f"(Precision: {results['precision'] * 100:.2f}%, Recall: {results['recall'] * 100:.2f}%)"
    )

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Public Benchmark Datasets (TVSum & SumMe)")
    parser.add_argument("--benchmark-dir", default="data/benchmark", help="Directory containing .h5 files")
    parser.add_argument("--method", choices=["visual_variance", "random", "uniform"], default="visual_variance")
    args = parser.parse_args()

    benchmark_dir = Path(args.benchmark_dir)
    tvsum_path = benchmark_dir / "eccv16_dataset_tvsum_google_pool5.h5"
    summe_path = benchmark_dir / "eccv16_dataset_summe_google_pool5.h5"

    print("=" * 65)
    print(" BENCHMARK EVALUATION PIPELINE (TVSum & SumMe)")
    print("=" * 65)

    all_results = []

    if tvsum_path.exists():
        res_tvsum = evaluate_single_dataset(tvsum_path, "TVSum", method=args.method, fscore_mode="mean")
        all_results.append(res_tvsum)
    else:
        print(f"[WARN] TVSum dataset not found at {tvsum_path}")

    if summe_path.exists():
        res_summe = evaluate_single_dataset(summe_path, "SumMe", method=args.method, fscore_mode="max")
        all_results.append(res_summe)
    else:
        print(f"[WARN] SumMe dataset not found at {summe_path}")

    print("\n" + "=" * 65)
    print(" BENCHMARK EVALUATION SUMMARY TABLE FOR REPORT")
    print("=" * 65)
    print(f"{'Dataset':<10} | {'Method':<16} | {'Kendall tau':<12} | {'Spearman rho':<12} | {'F1-Score':<10}")
    print("-" * 65)
    for r in all_results:
        print(
            f"{r['dataset']:<10} | {r['method']:<16} | {r['kendall_tau']:<12.4f} | "
            f"{r['spearman_rho']:<12.4f} | {r['f1_score'] * 100:<9.2f}%"
        )
    print("=" * 65)


if __name__ == "__main__":
    main()
