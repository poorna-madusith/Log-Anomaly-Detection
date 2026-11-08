"""
Thunderbird Detailed Evaluation Script (FYP Final Implementation)
========================================================================
Methodological Upgrades & Reviewer Bug Fixes:
1. Independent Validation Calibration: Finds optimal operating threshold tau* 
   strictly on the Validation set, then freezes it for Test evaluation.
2. Source-Chunk-Preserved Block-Bootstrap: Resamples ordered blocks within each sampled
   source chunk, without joining windows across sparse sampling gaps.
3. Two-class guards in bootstrap resampling.
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
    p = argparse.ArgumentParser(description="Deep Evaluation for Thunderbird Model")
    p.add_argument("--data_dir", type=str, default="./artifacts/Thunderbird/data")
    p.add_argument("--model_path", type=str, default="./artifacts/Thunderbird/checkpoints/best_model.pth")
    p.add_argument("--preds_file", type=str, default="./artifacts/Thunderbird/checkpoints/test_predictions.npz")
    p.add_argument("--block_size", type=int, default=200, help="Block size for bootstrap")
    p.add_argument("--n_boot", type=int, default=1000, help="Number of bootstrap resamples")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def safe_auroc(y, p):
    if len(np.unique(y)) < 2: return float("nan")
    return float(roc_auc_score(y, p))

def safe_auprc(y, p):
    if len(np.unique(y)) < 2: return float("nan")
    return float(average_precision_score(y, p))

def timeline_block_bootstrap_ci(
    y, probs, is_oov_window, block_size, n_boot, seed, chunk_ids=None
):
    """Bootstrap ordered blocks independently within each sampled source chunk.

    Thunderbird windows come from sparse byte offsets, so adjacent chunks in the
    NPZ are not adjacent in the original log timeline. Requiring chunk metadata
    prevents a bootstrap block from crossing those artificial boundaries.
    """
    if chunk_ids is None:
        raise ValueError(
            "Thunderbird test data is missing chunk_ids metadata. Re-run "
            "preprocess_tbird.py before calculating timeline bootstrap intervals."
        )
    rng = np.random.default_rng(seed)
    N = len(y)
    if len(probs) != N or len(is_oov_window) != N or len(chunk_ids) != N:
        raise ValueError("Labels, probabilities, OOV flags, and chunk IDs must align.")
    if N == 0 or block_size < 1 or n_boot < 1:
        raise ValueError("Bootstrap requires data and positive block_size/n_boot values.")

    # Preserve source order and require each source chunk to occupy one
    # contiguous run in the concatenated sampled dataset.
    chunk_runs = []
    run_start = 0
    for index in range(1, N + 1):
        if index == N or chunk_ids[index] != chunk_ids[run_start]:
            chunk_runs.append(np.arange(run_start, index, dtype=np.int64))
            run_start = index
    seen_chunk_ids = [chunk_ids[run[0]] for run in chunk_runs]
    if len(set(seen_chunk_ids)) != len(seen_chunk_ids):
        raise ValueError("Each Thunderbird chunk ID must occur in one contiguous run.")

    all_roc, all_prc = [], []
    oov_roc, oov_prc = [], []
    non_roc, non_prc = [], []

    for _ in range(n_boot):
        sampled_runs = []
        for run in chunk_runs:
            run_size = len(run)
            n_blocks = (run_size + block_size - 1) // block_size
            if run_size > block_size:
                valid_starts = np.arange(0, run_size - block_size + 1, dtype=np.int64)
            else:
                valid_starts = np.array([0], dtype=np.int64)
            chosen_starts = rng.choice(valid_starts, size=n_blocks, replace=True)
            run_indices = np.concatenate([
                np.arange(start, min(start + block_size, run_size), dtype=np.int64)
                for start in chosen_starts
            ])[:run_size]
            sampled_runs.append(run[run_indices])
        boot_idx = np.concatenate(sampled_runs)
        
        y_b = y[boot_idx]
        p_b = probs[boot_idx]
        oov_b = is_oov_window[boot_idx]

        # Overall
        v_roc = safe_auroc(y_b, p_b)
        v_prc = safe_auprc(y_b, p_b)
        if not np.isnan(v_roc): all_roc.append(v_roc)
        if not np.isnan(v_prc): all_prc.append(v_prc)

        # OOV subset within resampled timeline
        if np.sum(oov_b == 1) > 0 and len(np.unique(y_b[oov_b])) >= 2:
            v_roc_oov = safe_auroc(y_b[oov_b], p_b[oov_b])
            v_prc_oov = safe_auprc(y_b[oov_b], p_b[oov_b])
            if not np.isnan(v_roc_oov): oov_roc.append(v_roc_oov)
            if not np.isnan(v_prc_oov): oov_prc.append(v_prc_oov)

        # Non-OOV subset within resampled timeline
        if np.sum(~oov_b) > 0 and len(np.unique(y_b[~oov_b])) >= 2:
            v_roc_non = safe_auroc(y_b[~oov_b], p_b[~oov_b])
            v_prc_non = safe_auprc(y_b[~oov_b], p_b[~oov_b])
            if not np.isnan(v_roc_non): non_roc.append(v_roc_non)
            if not np.isnan(v_prc_non): non_prc.append(v_prc_non)

    def stats(arr, actual):
        if len(arr) == 0: return (actual, float("nan"), float("nan"))
        return (actual, np.percentile(arr, 2.5), np.percentile(arr, 97.5))

    return {
        "all_roc": stats(all_roc, safe_auroc(y, probs)),
        "all_prc": stats(all_prc, safe_auprc(y, probs)),
        "oov_roc": stats(oov_roc, safe_auroc(y[is_oov_window], probs[is_oov_window])),
        "oov_prc": stats(oov_prc, safe_auprc(y[is_oov_window], probs[is_oov_window])),
        "non_roc": stats(non_roc, safe_auroc(y[~is_oov_window], probs[~is_oov_window])),
        "non_prc": stats(non_prc, safe_auprc(y[~is_oov_window], probs[~is_oov_window])),
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
        print(f"Error: {args.preds_file} not found. Run train_tbird.py first.")
        sys.exit(1)
        
    model_mtime = os.path.getmtime(args.model_path) if os.path.exists(args.model_path) else 0
    preds_mtime = os.path.getmtime(args.preds_file) if os.path.exists(args.preds_file) else 0
    if model_mtime > preds_mtime:
        print(f"[!] WARNING: {args.preds_file} is older than {args.model_path}.")
        print("    Predictions may be desynchronized from the latest model checkpoint.")
        
    print("\n" + "="*72)
    print("THUNDERBIRD DEEP EVALUATION & STRATIFICATION REPORT (Reviewer-Vetted)")
    print("="*72)

    with np.load(args.preds_file) as preds:
        y, probs = np.asarray(preds["targets"]), np.asarray(preds["probs"])
    
    with np.load(os.path.join(args.data_dir, "test_data.npz")) as test_data:
        X_struct = test_data["X_struct"]
        if "chunk_ids" not in test_data or "chunk_window_indices" not in test_data:
            raise ValueError(
                "Thunderbird test_data.npz lacks source chunk metadata. "
                "Re-run preprocessing with preprocess_tbird.py before evaluation."
            )
        chunk_ids = test_data["chunk_ids"]
        chunk_window_indices = test_data["chunk_window_indices"]
        source_y = np.asarray(test_data["y"])
    if len(y) != len(source_y) or not np.array_equal(y, source_y):
        raise ValueError(
            "Thunderbird predictions do not align with test_data.npz labels. "
            "Regenerate predictions with the current checkpoint and preprocessing artifacts."
        )
    if len(chunk_window_indices) != len(X_struct):
        raise ValueError("Thunderbird chunk window metadata does not match test rows.")
    if np.any(chunk_window_indices < 0):
        raise ValueError("Thunderbird chunk window indices must be non-negative.")
    source_keys = list(zip(chunk_ids.tolist(), chunk_window_indices.tolist()))
    if len(set(source_keys)) != len(source_keys):
        raise ValueError("Thunderbird test data contains duplicate source window identities.")
    if source_keys != sorted(source_keys):
        raise ValueError("Thunderbird test windows are not in source timeline order.")
    # Unified convention: ID 1 is strictly <OOV>
    is_oov_window = (X_struct == 1).any(axis=1)
    
    print(f"[*] Running source-chunk-preserved block-bootstrap ({args.n_boot} resamples, block={args.block_size})...")
    res = timeline_block_bootstrap_ci(
        y, probs, is_oov_window, args.block_size, args.n_boot, args.seed,
        chunk_ids=chunk_ids,
    )
    
    print(f"\n[Overall Test Set] (N={len(y):,}, Anomalies={int(y.sum()):,})")
    print(f"  AUROC: {res['all_roc'][0]:.4f}  [95% CI: {res['all_roc'][1]:.4f} - {res['all_roc'][2]:.4f}]")
    print(f"  AUPRC: {res['all_prc'][0]:.4f}  [95% CI: {res['all_prc'][1]:.4f} - {res['all_prc'][2]:.4f}]")
    
    n_oov = np.sum(is_oov_window)
    print(f"\n[OOV Subset (Unseen Data)] (N={n_oov:,}, Anomalies={int(y[is_oov_window].sum()):,})")
    print(f"  AUROC: {res['oov_roc'][0]:.4f}  [95% CI: {res['oov_roc'][1]:.4f} - {res['oov_roc'][2]:.4f}]")
    print(f"  AUPRC: {res['oov_prc'][0]:.4f}  [95% CI: {res['oov_prc'][1]:.4f} - {res['oov_prc'][2]:.4f}]")
    
    n_non = np.sum(~is_oov_window)
    print(f"\n[Non-OOV Subset (Seen Data)] (N={n_non:,}, Anomalies={int(y[~is_oov_window].sum()):,})")
    print(f"  AUROC: {res['non_roc'][0]:.4f}  [95% CI: {res['non_roc'][1]:.4f} - {res['non_roc'][2]:.4f}]")
    print(f"  AUPRC: {res['non_prc'][0]:.4f}  [95% CI: {res['non_prc'][1]:.4f} - {res['non_prc'][2]:.4f}]")

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
