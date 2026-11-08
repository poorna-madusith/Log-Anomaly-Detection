"""
Thesis Visualization Generator (FYP Final Implementation)
========================================================================
Generates publication-quality, high-resolution figures (300 DPI) for:
1. Combined ROC Curves across HDFS, BGL, and Thunderbird.
2. Combined Precision-Recall (PR) Curves across all three datasets.
3. BGL Out-of-Vocabulary (OOV) Stratification Analysis (Seen vs. Unseen).
4. Master Benchmark Summary Comparison (AUROC, AUPRC, F1) computed dynamically.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score, f1_score

def set_academic_style():
    """Sets clean, publication-ready plot aesthetics."""
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'sans-serif',
        'axes.labelsize': 14,
        'axes.titlesize': 15,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 16,
        'lines.linewidth': 2.2,
        'grid.alpha': 0.35,
        'grid.linestyle': '--'
    })

def compute_val_calibrated_f1(val_preds_path, test_y, test_p, dataset_name):
    """Computes test F1 score strictly using the operating threshold calibrated on validation data."""
    if not os.path.exists(val_preds_path):
        raise FileNotFoundError(
            f"{dataset_name}: validation predictions not found at {val_preds_path}. "
            "Run the dataset training or evaluation script before generating plots."
        )
    with np.load(val_preds_path) as v:
        val_y, val_p = v["targets"], v["probs"]
    if len(np.unique(val_y)) < 2:
        raise ValueError(f"{dataset_name}: validation labels must contain both normal and anomalous examples.")
    prec_v, rec_v, thresh_v = precision_recall_curve(val_y, val_p)
    f1_v = 2 * (prec_v * rec_v) / (prec_v + rec_v + 1e-9)
    best_idx = np.argmax(f1_v)
    tau = float(thresh_v[best_idx]) if best_idx < len(thresh_v) else 0.5
    test_pred = (test_p >= tau).astype(int)
    test_f1 = float(f1_score(test_y, test_pred, zero_division=0))
    print(f"  [{dataset_name}] Frozen Val Threshold tau* = {tau:.4f} -> Test F1 = {test_f1:.4f}")
    return test_f1

def main():
    set_academic_style()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, "artifacts", "figures")
    os.makedirs(output_dir, exist_ok=True)
    
    print("[*] Loading test predictions from all three datasets...")
    with np.load(os.path.join(base_dir, "artifacts", "HDFS", "checkpoints", "test_predictions.npz")) as f:
        y_hdfs, p_hdfs = f["targets"], f["probs"]
    with np.load(os.path.join(base_dir, "artifacts", "BGL", "checkpoints", "test_predictions.npz")) as f:
        y_bgl, p_bgl = f["targets"], f["probs"]
    with np.load(os.path.join(base_dir, "artifacts", "Thunderbird", "checkpoints", "test_predictions.npz")) as f:
        y_tbird, p_tbird = f["targets"], f["probs"]

    datasets = [
        {"name": "HDFS (Cloud Storage)", "y": y_hdfs, "p": p_hdfs, "color": "#1f77b4"},
        {"name": "BGL (Supercomputer)", "y": y_bgl, "p": p_bgl, "color": "#ff7f0e"},
        {"name": "Thunderbird (30GB Chunked)", "y": y_tbird, "p": p_tbird, "color": "#2ca02c"}
    ]

    # ---------------------------------------------------------
    # 1. Combined ROC Curves
    # ---------------------------------------------------------
    print("[*] Generating Combined ROC Curve...")
    plt.figure(figsize=(8, 6.5))
    for d in datasets:
        fpr, tpr, _ = roc_curve(d["y"], d["p"])
        score = roc_auc_score(d["y"], d["p"])
        plt.plot(fpr, tpr, label=f"{d['name']} (AUROC = {score:.4f})", color=d["color"])
        
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Classifier Reference (AUROC = 0.5000)")
    plt.xlim([-0.01, 1.01])
    plt.ylim([0.0, 1.02])
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR / Recall)")
    plt.title("Receiver Operating Characteristic (ROC) Across Benchmarks")
    plt.legend(loc="lower right", frameon=True, framealpha=0.9)
    plt.grid(True)
    plt.tight_layout()
    roc_path = os.path.join(output_dir, "figure_1_roc_curves.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"  [+] Saved: {roc_path}")

    # ---------------------------------------------------------
    # 2. Combined Precision-Recall (PR) Curves
    # ---------------------------------------------------------
    print("[*] Generating Combined Precision-Recall Curve...")
    plt.figure(figsize=(8, 6.5))
    for d in datasets:
        precision, recall, _ = precision_recall_curve(d["y"], d["p"])
        score = average_precision_score(d["y"], d["p"])
        plt.plot(recall, precision, label=f"{d['name']} (AUPRC = {score:.4f})", color=d["color"])

    plt.xlim([-0.01, 1.01])
    plt.ylim([0.0, 1.02])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall (PR) Curves Across Benchmarks")
    plt.legend(loc="lower left", frameon=True, framealpha=0.9)
    plt.grid(True)
    plt.tight_layout()
    pr_path = os.path.join(output_dir, "figure_2_pr_curves.png")
    plt.savefig(pr_path, dpi=300)
    plt.close()
    print(f"  [+] Saved: {pr_path}")

    # ---------------------------------------------------------
    # 3. BGL Out-of-Vocabulary (OOV) Stratification
    # ---------------------------------------------------------
    print("[*] Generating BGL OOV Stratification Plot (Methodologically Vetted)...")
    with np.load(os.path.join(base_dir, "artifacts", "BGL", "data", "test_data.npz")) as bgl_test_data:
        # Unified convention: ID 1 is strictly <OOV>
        is_oov = (bgl_test_data["X_struct"] == 1).any(axis=1)

    plt.figure(figsize=(8, 6.5))
    
    # Seen Data (Non-OOV)
    fpr_seen, tpr_seen, _ = roc_curve(y_bgl[~is_oov], p_bgl[~is_oov])
    auc_seen = roc_auc_score(y_bgl[~is_oov], p_bgl[~is_oov])
    plt.plot(fpr_seen, tpr_seen, label=f"Seen Templates / Non-OOV (AUROC = {auc_seen:.4f})", color="#2ca02c")

    # Overall Test Set
    fpr_all, tpr_all, _ = roc_curve(y_bgl, p_bgl)
    auc_all = roc_auc_score(y_bgl, p_bgl)
    plt.plot(fpr_all, tpr_all, label=f"Overall Test Set (AUROC = {auc_all:.4f})", color="#1f77b4", linestyle="--")

    # Unseen Data (OOV)
    fpr_oov, tpr_oov, _ = roc_curve(y_bgl[is_oov], p_bgl[is_oov])
    auc_oov = roc_auc_score(y_bgl[is_oov], p_bgl[is_oov])
    plt.plot(fpr_oov, tpr_oov, label=f"Unseen Templates / OOV Drift (AUROC = {auc_oov:.4f})", color="#d62728")

    # Honest reference: Standard theoretical uninformative classifier line
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random Classifier Reference (AUROC = 0.5000)")
    plt.xlim([-0.01, 1.01])
    plt.ylim([0.0, 1.02])
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR / Recall)")
    plt.title("BGL Log Drift: Resilience on Unseen (OOV) Templates")
    plt.legend(loc="lower right", frameon=True, framealpha=0.9)
    plt.grid(True)
    plt.tight_layout()
    oov_path = os.path.join(output_dir, "figure_3_bgl_oov_stratification.png")
    plt.savefig(oov_path, dpi=300)
    plt.close()
    print(f"  [+] Saved: {oov_path}")

    # ---------------------------------------------------------
    # 4. Master Benchmark Summary Bar Chart (Validation-Calibrated)
    # ---------------------------------------------------------
    print("[*] Generating Benchmark Comparison Bar Chart (Validation-Calibrated F1)...")
    metrics_names = ["AUROC", "AUPRC", "F1 Score\n(Val-Calibrated)"]
    
    hdfs_val_path = os.path.join(base_dir, "artifacts", "HDFS", "checkpoints", "val_predictions.npz")
    bgl_val_path = os.path.join(base_dir, "artifacts", "BGL", "checkpoints", "val_predictions.npz")
    tbird_val_path = os.path.join(base_dir, "artifacts", "Thunderbird", "checkpoints", "val_predictions.npz")

    scores_hdfs = [
        float(roc_auc_score(y_hdfs, p_hdfs)),
        float(average_precision_score(y_hdfs, p_hdfs)),
        compute_val_calibrated_f1(hdfs_val_path, y_hdfs, p_hdfs, "HDFS")
    ]
    scores_bgl = [
        float(roc_auc_score(y_bgl, p_bgl)),
        float(average_precision_score(y_bgl, p_bgl)),
        compute_val_calibrated_f1(bgl_val_path, y_bgl, p_bgl, "BGL")
    ]
    scores_tbird = [
        float(roc_auc_score(y_tbird, p_tbird)),
        float(average_precision_score(y_tbird, p_tbird)),
        compute_val_calibrated_f1(tbird_val_path, y_tbird, p_tbird, "Thunderbird")
    ]

    x = np.arange(len(metrics_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 6))
    rects1 = ax.bar(x - width, scores_hdfs, width, label='HDFS', color='#1f77b4', edgecolor='black', alpha=0.85)
    rects2 = ax.bar(x, scores_bgl, width, label='BGL', color='#ff7f0e', edgecolor='black', alpha=0.85)
    rects3 = ax.bar(x + width, scores_tbird, width, label='Thunderbird', color='#2ca02c', edgecolor='black', alpha=0.85)

    ax.set_ylabel('Score')
    ax.set_title('Cross-Dataset Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.set_ylim(0.0, 1.10)
    ax.legend(loc='lower left', frameon=True)
    ax.grid(axis='y', alpha=0.4)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    fig.tight_layout()
    bar_path = os.path.join(output_dir, "figure_4_benchmark_comparison.png")
    plt.savefig(bar_path, dpi=300)
    plt.close()
    print(f"  [+] Saved: {bar_path}")

    print(f"\n[+] All 4 thesis figures successfully updated at 300 DPI in {output_dir}!")

if __name__ == "__main__":
    main()
