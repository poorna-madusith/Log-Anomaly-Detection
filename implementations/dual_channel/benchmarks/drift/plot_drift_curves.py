"""
Publication Plot Generator: 0%-30% Drift Robustness Curves (Figure 5)
========================================================================
Generates publication-quality 300-DPI side-by-side curves:
- Left: Anomaly F1 Score vs. Drift Severity (0%, 10%, 20%, 30%)
- Right: AUPRC vs. Drift Severity (0%, 10%, 20%, 30%)
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

def set_academic_style():
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'sans-serif',
        'axes.labelsize': 14,
        'axes.titlesize': 15,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        'figure.titlesize': 16,
        'lines.linewidth': 2.5,
        'grid.alpha': 0.35,
        'grid.linestyle': '--'
    })

def main():
    parser = argparse.ArgumentParser(description="Plot controlled-drift results for one dataset")
    parser.add_argument("--dataset", choices=["HDFS", "BGL", "Thunderbird"], default="HDFS")
    args = parser.parse_args()

    set_academic_style()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(os.path.dirname(current_dir))
    dataset_dir = os.path.join(root_dir, "artifacts", args.dataset)
    drift_dir = os.path.join(dataset_dir, "drift")
    json_path = os.path.join(drift_dir, "drift_benchmark_results.json")
    
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found. Run run_drift_benchmark.py first.")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict) or payload.get("dataset") != args.dataset:
        raise SystemExit(
            f"Expected a dataset-tagged drift result for {args.dataset} at {json_path}; "
            "rerun run_drift_benchmark.py for this dataset."
        )
    data = payload.get("results")
    if not isinstance(data, list) or not data:
        raise SystemExit(f"No drift results found in {json_path}; rerun the benchmark.")

    tiers = [d["drift_tier"] for d in data]
    x = np.arange(len(tiers))

    # Extract metrics
    st_f1 = [d["structural_only"]["pos_f1"] for d in data]
    se_f1 = [d["semantic_only"]["pos_f1"] for d in data]
    du_f1 = [d["dual_channel"]["pos_f1"] for d in data]

    st_auprc = [d["structural_only"]["auprc"] for d in data]
    se_auprc = [d["semantic_only"]["auprc"] for d in data]
    du_auprc = [d["dual_channel"]["auprc"] for d in data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # 1. Left Plot: F1-Score
    ax1.plot(x, du_f1, 'o-', color='#1f77b4', label='Dual-Channel Fusion (Ours)', markersize=8)
    ax1.plot(x, se_f1, 's--', color='#ff7f0e', label='Semantic-Only (Ablation)', markersize=8)
    ax1.plot(x, st_f1, '^:', color='#d62728', label='Structural-Only (DeepLog)', markersize=8)
    
    ax1.set_xlabel("Simulated CI/CD Drift Ratio")
    ax1.set_ylabel("Validation-Calibrated F1 Score")
    ax1.set_title("Operational F1 Score under Drift")
    ax1.set_xticks(x)
    ax1.set_xticklabels(tiers)
    ax1.set_ylim(0.15, 1.02)
    ax1.grid(True)
    ax1.legend(loc='lower left', frameon=True, framealpha=0.9)

    # 2. Right Plot: AUPRC
    ax2.plot(x, du_auprc, 'o-', color='#1f77b4', label='Dual-Channel Fusion (Ours)', markersize=8)
    ax2.plot(x, se_auprc, 's--', color='#ff7f0e', label='Semantic-Only (Ablation)', markersize=8)
    ax2.plot(x, st_auprc, '^:', color='#d62728', label='Structural-Only (DeepLog)', markersize=8)

    ax2.set_xlabel("Simulated CI/CD Drift Ratio")
    ax2.set_ylabel("Precision-Recall AUC (AUPRC)")
    ax2.set_title("AUPRC Ranking under Drift")
    ax2.set_xticks(x)
    ax2.set_xticklabels(tiers)
    ax2.set_ylim(0.35, 1.02)
    ax2.grid(True)
    ax2.legend(loc='lower left', frameon=True, framealpha=0.9)

    plt.suptitle(f"Architectural Robustness against Incremental CI/CD Log Drift ({args.dataset})", y=1.02)
    plt.tight_layout()

    # Save to both local results and master figures directory
    out_local = os.path.join(drift_dir, "figure_5_drift_robustness_curves.png")
    out_master = os.path.join(root_dir, "artifacts", "figures", f"figure_5_drift_robustness_curves_{args.dataset}.png")
    os.makedirs(os.path.dirname(out_local), exist_ok=True)
    os.makedirs(os.path.dirname(out_master), exist_ok=True)
    
    plt.savefig(out_local, dpi=300, bbox_inches="tight")
    plt.savefig(out_master, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"[+] Saved Figure 5 to: {out_local}")
    print(f"[+] Saved Figure 5 to: {out_master}")

if __name__ == "__main__":
    main()
