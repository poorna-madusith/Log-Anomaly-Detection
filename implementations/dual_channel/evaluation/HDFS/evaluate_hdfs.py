"""
HDFS Detailed Evaluation Script (FYP Final Implementation)
========================================================================
Methodological Upgrades & Reviewer Bug Fixes:
1. Independent Validation Calibration: Finds optimal operating threshold tau* 
   strictly on the Validation set, then freezes it for Test evaluation (no data snooping).
2. Two-class guards in bootstrap resampling.
3. Reports both Validation-Calibrated Test F1 and Reference Test-Tuned Oracle F1.
"""

import os
import sys
import argparse
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, precision_score, recall_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from models.model import DualChannelBiLSTM
from models.dataset import create_dataloader

def parse_args():
    p = argparse.ArgumentParser(description="Deep Evaluation for HDFS Model")
    p.add_argument("--data_dir", type=str, default="./artifacts/HDFS/data")
    p.add_argument("--model_path", type=str, default="./artifacts/HDFS/checkpoints/best_model.pth")
    p.add_argument("--preds_file", type=str, default="./artifacts/HDFS/checkpoints/test_predictions.npz")
    p.add_argument("--block_size", type=int, default=1000, help="Block size for bootstrap (HDFS has 115k test blocks)")
    p.add_argument("--n_boot", type=int, default=1000, help="Number of bootstrap resamples")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def safe_auroc(y, p):
    if len(np.unique(y)) < 2: return float("nan")
    return float(roc_auc_score(y, p))

def safe_auprc(y, p):
    if len(np.unique(y)) < 2: return float("nan")
    return float(average_precision_score(y, p))

def block_bootstrap_ci(y, probs, block_size, n_boot, seed):
    rng = np.random.default_rng(seed)
    N = len(y)
    starts = list(range(0, N, block_size))
    n_blocks = len(starts)

    auroc_samples, auprc_samples = [], []
    for _ in range(n_boot):
        chosen = rng.choice(n_blocks, size=n_blocks, replace=True)
        # Circular indexing guarantees every block has exact block_size elements
        boot_idx = np.concatenate([(starts[c] + np.arange(block_size)) % N for c in chosen])[:N]
        y_b = y[boot_idx]
        p_b = probs[boot_idx]
        
        v_roc = safe_auroc(y_b, p_b)
        v_prc = safe_auprc(y_b, p_b)
        if not np.isnan(v_roc): auroc_samples.append(v_roc)
        if not np.isnan(v_prc): auprc_samples.append(v_prc)

    def stats(samples, actual):
        if not samples:
            return (actual, float("nan"), float("nan"))
        return (actual, float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5)))

    return {
        "auroc": stats(auroc_samples, safe_auroc(y, probs)),
        "auprc": stats(auprc_samples, safe_auprc(y, probs))
    }

def get_or_create_val_predictions(args):
    val_preds_path = os.path.join(os.path.dirname(args.preds_file), "val_predictions.npz")
    model_mtime = os.path.getmtime(args.model_path) if os.path.exists(args.model_path) else 0
    val_mtime = os.path.getmtime(val_preds_path) if os.path.exists(val_preds_path) else 0
    
    # Only reuse cache if it is at least as fresh as the model checkpoint
    if os.path.exists(val_preds_path) and val_mtime >= model_mtime:
        data = np.load(val_preds_path)
        return data["targets"], data["probs"]
        
    print("[*] Generating Validation Set predictions using best_model.pth...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(args.model_path, map_location=device, weights_only=True)
    vocab_size = state_dict["struct_embedding.weight"].shape[0]
    model = DualChannelBiLSTM(vocab_size=vocab_size).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    
    val_loader, _ = create_dataloader(
        os.path.join(args.data_dir, "val_data.npz"), 
        os.path.join(args.data_dir, "string_matrix.npy"), 
        batch_size=256, shuffle=False
    )
    
    val_y, val_p = [], []
    with torch.no_grad():
        for struct_seq, sem_seq, y in val_loader:
            logits, _ = model(struct_seq.to(device), sem_seq.to(device))
            probs = torch.sigmoid(logits).cpu().numpy()
            val_p.extend(probs)
            val_y.extend(y.numpy())
            
    val_y = np.array(val_y)
    val_p = np.array(val_p)
    np.savez_compressed(val_preds_path, targets=val_y, probs=val_p)
    print(f"[+] Saved validation predictions to {val_preds_path}")
    return val_y, val_p

def main():
    args = parse_args()
    if not os.path.exists(args.preds_file):
        print(f"Error: {args.preds_file} not found. Run train_hdfs.py first.")
        sys.exit(1)
        
    model_mtime = os.path.getmtime(args.model_path) if os.path.exists(args.model_path) else 0
    preds_mtime = os.path.getmtime(args.preds_file) if os.path.exists(args.preds_file) else 0
    if model_mtime > preds_mtime:
        print(f"[!] WARNING: {args.preds_file} is older than {args.model_path}.")
        print("    Predictions may be desynchronized from the latest model checkpoint.")
        
    print("\n" + "="*72)
    print("HDFS DEEP EVALUATION REPORT (Reviewer-Vetted)")
    print("="*72)

    preds = np.load(args.preds_file)
    y, probs = preds["targets"], preds["probs"]
    
    print(f"[*] Running block-bootstrap ({args.n_boot} resamples, block={args.block_size})...")
    stats = block_bootstrap_ci(y, probs, args.block_size, args.n_boot, args.seed)
    
    print(f"\n[Overall Test Set] (N={len(y):,}, Anomalies={int(y.sum()):,})")
    print(f"  AUROC: {stats['auroc'][0]:.4f}  [95% CI: {stats['auroc'][1]:.4f} - {stats['auroc'][2]:.4f}]")
    print(f"  AUPRC: {stats['auprc'][0]:.4f}  [95% CI: {stats['auprc'][1]:.4f} - {stats['auprc'][2]:.4f}]")

    print("\n" + "="*72)
    print("OPERATING THRESHOLD CALIBRATION (Strict Validation Tuning)")
    print("="*72)
    val_y, val_probs = get_or_create_val_predictions(args)
    prec_v, rec_v, thresh_v = precision_recall_curve(val_y, val_probs)
    f1_v = 2 * (prec_v * rec_v) / (prec_v + rec_v + 1e-9)
    best_v_idx = np.argmax(f1_v)
    calibrated_thresh = thresh_v[best_v_idx] if best_v_idx < len(thresh_v) else 0.5
    
    print(f"[+] Optimal Validation Threshold: tau* = {calibrated_thresh:.4f} (Validation F1 = {f1_v[best_v_idx]:.4f})")
    
    test_preds_cal = (probs >= calibrated_thresh).astype(int)
    test_p = precision_score(y, test_preds_cal, zero_division=0)
    test_r = recall_score(y, test_preds_cal, zero_division=0)
    test_f1 = f1_score(y, test_preds_cal, zero_division=0)
    
    print(f"\n[Frozen Operating Threshold on Test Set] (tau = {calibrated_thresh:.4f})")
    print(f"  Precision: {test_p:.4f}")
    print(f"  Recall:    {test_r:.4f}")
    print(f"  F1 Score:  {test_f1:.4f}")
    
    prec_t, rec_t, thresh_t = precision_recall_curve(y, probs)
    f1_t = 2 * (prec_t * rec_t) / (prec_t + rec_t + 1e-9)
    best_t_idx = np.argmax(f1_t)
    oracle_thresh = thresh_t[best_t_idx] if best_t_idx < len(thresh_t) else thresh_t[-1]
    print(f"\n(Reference Oracle Test-Tuned F1: {f1_t[best_t_idx]:.4f} at tau = {oracle_thresh:.4f})")
    print("="*72)

if __name__ == "__main__":
    main()
