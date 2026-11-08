"""
Master Baseline Training & Channel Ablation Benchmark
========================================================================
Performs a rigorous, controlled 3-way ablation study on the exact same data split:
1. Structural-Only Baseline (DeepLog / Drain-LSTM equivalent): Channel 1 in isolation
2. Semantic-Only Baseline (Ablation model): Channel 2 in isolation
3. Dual-Channel Fusion Model (Proposed): Channel 1 + Channel 2 + Structural Dropout

Evaluates all models using identical methodology:
- Frozen validation threshold calibration (tau*)
- Real-world OOV drift stratification (Seen vs. Unseen templates)
- Saves checkpoints and comparative JSON summary
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, precision_score, recall_score, f1_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
from models.dataset import create_dataloader
from models.baseline_models import StructuralBaselineBiLSTM, SemanticBaselineBiLSTM

def parse_args():
    p = argparse.ArgumentParser(description="Run Baseline & Channel Ablation Experiments")
    p.add_argument("--dataset", type=str, default="BGL", choices=["BGL", "HDFS", "Thunderbird"],
                   help="Dataset to benchmark baselines on (Default: BGL for temporal drift)")
    p.add_argument("--epochs", type=int, default=10, help="Training epochs per baseline")
    p.add_argument("--batch_size", type=int, default=256, help="Batch size")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    p.add_argument("--seed", type=int, default=42, help="Reproducibility seed")
    return p.parse_args()

def evaluate_model(model, loader, device, is_semantic=False):
    model.eval()
    all_y, all_p = [], []
    with torch.no_grad():
        for struct_seq, sem_seq, y in loader:
            struct_seq = struct_seq.to(device)
            sem_seq = sem_seq.to(device)
            if is_semantic:
                logits, _ = model(struct_seq, sem_seq)
            else:
                logits, _ = model(struct_seq)
            probs = torch.sigmoid(logits)
            all_y.extend(y.cpu().numpy())
            all_p.extend(probs.cpu().numpy())
            
    all_y = np.array(all_y)
    all_p = np.array(all_p)
    auroc = float(roc_auc_score(all_y, all_p)) if len(np.unique(all_y)) >= 2 else float("nan")
    auprc = float(average_precision_score(all_y, all_p)) if len(np.unique(all_y)) >= 2 else float("nan")
    return auroc, auprc, all_y, all_p

def train_and_eval_baseline(model_name, model, train_loader, val_loader, test_loader, 
                            pos_weight, args, device, is_semantic=False):
    print("\n" + "="*70)
    print(f" TRAINING: {model_name}")
    print("="*70)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    save_dir = os.path.join(BASE_DIR, "artifacts", args.dataset, "checkpoints")
    checkpoint_file = os.path.join(save_dir, f"baseline_{'semantic' if is_semantic else 'structural'}.pth")
    
    best_val_auprc = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for struct_seq, sem_seq, y in pbar:
            struct_seq = struct_seq.to(device)
            sem_seq = sem_seq.to(device)
            y = y.to(device)
            
            optimizer.zero_grad()
            if is_semantic:
                logits, _ = model(struct_seq, sem_seq)
            else:
                logits, _ = model(struct_seq)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        val_auroc, val_auprc, _, _ = evaluate_model(model, val_loader, device, is_semantic)
        print(f"  --> Epoch {epoch}: Val AUROC = {val_auroc:.4f} | Val AUPRC = {val_auprc:.4f}")
        
        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            torch.save(model.state_dict(), checkpoint_file)
            print(f"  [+] Saved new best {model_name} (Val AUPRC: {best_val_auprc:.4f})")
            
    # Load best checkpoint for validation threshold calibration & test evaluation
    model.load_state_dict(torch.load(checkpoint_file, map_location=device, weights_only=True))
    _, _, val_y, val_probs = evaluate_model(model, val_loader, device, is_semantic)
    test_auroc, test_auprc, test_y, test_probs = evaluate_model(model, test_loader, device, is_semantic)
    
    # Validation threshold tuning
    prec_v, rec_v, thresh_v = precision_recall_curve(val_y, val_probs)
    f1_v = 2 * (prec_v * rec_v) / (prec_v + rec_v + 1e-9)
    best_v_idx = np.argmax(f1_v)
    tau = float(thresh_v[best_v_idx]) if best_v_idx < len(thresh_v) else 0.5
    
    # Frozen test evaluation
    test_preds = (test_probs >= tau).astype(int)
    test_p = float(precision_score(test_y, test_preds, zero_division=0))
    test_r = float(recall_score(test_y, test_preds, zero_division=0))
    test_f1 = float(f1_score(test_y, test_preds, zero_division=0))
    
    return {
        "auroc": test_auroc,
        "auprc": test_auprc,
        "precision": test_p,
        "recall": test_r,
        "f1": test_f1,
        "tau": tau,
        "probs": test_probs
    }

def main():
    args = parse_args()
    base_dir = BASE_DIR
    data_dir = os.path.join(base_dir, "artifacts", args.dataset, "data")
    save_dir = os.path.join(base_dir, "artifacts", args.dataset, "checkpoints")
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\n" + "#"*72)
    print(f" CONTROLLED CHANNEL ABLATION & BASELINE BENCHMARK ({args.dataset})")
    print(f" Device: {device} | Epochs: {args.epochs} | Seed: {args.seed}")
    print("#"*72)
    
    # 1. Load Data
    train_loader, train_ds = create_dataloader(
        os.path.join(data_dir, "train_data.npz"),
        os.path.join(data_dir, "string_matrix.npy"),
        args.batch_size, shuffle=True
    )
    val_loader, _ = create_dataloader(
        os.path.join(data_dir, "val_data.npz"),
        os.path.join(data_dir, "string_matrix.npy"),
        args.batch_size, shuffle=False
    )
    test_loader, _ = create_dataloader(
        os.path.join(data_dir, "test_data.npz"),
        os.path.join(data_dir, "string_matrix.npy"),
        args.batch_size, shuffle=False
    )
    pos_weight = train_ds.get_pos_weight().to(device)
    
    # Determine vocabulary size from Dual-Channel checkpoint for parity
    dual_ckpt = os.path.join(save_dir, "best_model.pth")
    dual_state = torch.load(dual_ckpt, map_location=device, weights_only=True)
    vocab_size = dual_state["struct_embedding.weight"].shape[0]
    
    # 2. Train Structural-Only Baseline (DeepLog Equivalent)
    struct_model = StructuralBaselineBiLSTM(vocab_size=vocab_size).to(device)
    struct_res = train_and_eval_baseline(
        "Structural-Only Baseline (DeepLog Equivalent)",
        struct_model, train_loader, val_loader, test_loader, pos_weight, args, device, is_semantic=False
    )
    
    # 3. Train Semantic-Only Baseline (Semantic Ablation)
    sem_model = SemanticBaselineBiLSTM(sem_input_dim=384).to(device)
    sem_res = train_and_eval_baseline(
        "Semantic-Only Baseline (Semantic Ablation)",
        sem_model, train_loader, val_loader, test_loader, pos_weight, args, device, is_semantic=True
    )
    
    # 4. Load Saved Dual-Channel Predictions for Direct Parity
    with np.load(os.path.join(save_dir, "test_predictions.npz")) as f:
        dual_probs = f["probs"]
        y_test = f["targets"]
    with np.load(os.path.join(save_dir, "val_predictions.npz")) as f:
        val_probs = f["probs"]
        val_y = f["targets"]
        
    prec_v, rec_v, thresh_v = precision_recall_curve(val_y, val_probs)
    f1_v = 2 * (prec_v * rec_v) / (prec_v + rec_v + 1e-9)
    best_v_idx = np.argmax(f1_v)
    dual_tau = float(thresh_v[best_v_idx]) if best_v_idx < len(thresh_v) else 0.5
    dual_preds = (dual_probs >= dual_tau).astype(int)
    
    dual_res = {
        "auroc": float(roc_auc_score(y_test, dual_probs)),
        "auprc": float(average_precision_score(y_test, dual_probs)),
        "precision": float(precision_score(y_test, dual_preds, zero_division=0)),
        "recall": float(recall_score(y_test, dual_preds, zero_division=0)),
        "f1": float(f1_score(y_test, dual_preds, zero_division=0)),
        "tau": dual_tau,
        "probs": dual_probs
    }
    
    # 5. OOV Stratification Analysis (if BGL or Thunderbird)
    with np.load(os.path.join(data_dir, "test_data.npz")) as td:
        is_oov = (td["X_struct"] == 1).any(axis=1) if "X_struct" in td else None
        
    has_oov = is_oov is not None and is_oov.sum() > 0 and len(np.unique(y_test[is_oov])) >= 2
    
    def get_oov_breakdown(res_dict):
        if not has_oov: return ("N/A", "N/A", "N/A", "N/A")
        p = res_dict["probs"]
        oov_roc = roc_auc_score(y_test[is_oov], p[is_oov])
        oov_prc = average_precision_score(y_test[is_oov], p[is_oov])
        seen_roc = roc_auc_score(y_test[~is_oov], p[~is_oov])
        seen_prc = average_precision_score(y_test[~is_oov], p[~is_oov])
        return (f"{seen_roc:.4f}", f"{seen_prc:.4f}", f"{oov_roc:.4f}", f"{oov_prc:.4f}")

    # Compute breakdown for all three
    st_seen_roc, st_seen_prc, st_oov_roc, st_oov_prc = get_oov_breakdown(struct_res)
    se_seen_roc, se_seen_prc, se_oov_roc, se_oov_prc = get_oov_breakdown(sem_res)
    du_seen_roc, du_seen_prc, du_oov_roc, du_oov_prc = get_oov_breakdown(dual_res)

    # 6. Print Master Ablation Summary Table
    print("\n" + "="*85)
    print(f" MASTER 3-WAY ARCHITECTURAL ABLATION RESULTS ({args.dataset} TEST SET)")
    print("="*85)
    header = f"{'Model Architecture':<32} | {'AUROC':<7} | {'AUPRC':<7} | {'F1 (val)':<8} | {'Recall':<7} | {'OOV AUPRC':<9}"
    print(header)
    print("-" * len(header))
    print(f"{'1. Structural-Only (DeepLog)':<32} | {struct_res['auroc']:.4f}  | {struct_res['auprc']:.4f}  | {struct_res['f1']:.4f}   | {struct_res['recall']:.4f}  | {st_oov_prc:<9}")
    print(f"{'2. Semantic-Only (Ablation)':<32} | {sem_res['auroc']:.4f}  | {sem_res['auprc']:.4f}  | {sem_res['f1']:.4f}   | {sem_res['recall']:.4f}  | {se_oov_prc:<9}")
    print(f"{'3. Dual-Channel Fusion (Ours)':<32} | {dual_res['auroc']:.4f}  | {dual_res['auprc']:.4f}  | {dual_res['f1']:.4f}   | {dual_res['recall']:.4f}  | {du_oov_prc:<9}")
    print("="*85)
    
    # Save results to JSON
    out_json = os.path.join(save_dir, "ablation_benchmark_results.json")
    results_payload = {
        "dataset": args.dataset,
        "structural_only": {
            "auroc": struct_res["auroc"], "auprc": struct_res["auprc"],
            "f1": struct_res["f1"], "precision": struct_res["precision"], "recall": struct_res["recall"],
            "oov_auprc": st_oov_prc, "seen_auprc": st_seen_prc
        },
        "semantic_only": {
            "auroc": sem_res["auroc"], "auprc": sem_res["auprc"],
            "f1": sem_res["f1"], "precision": sem_res["precision"], "recall": sem_res["recall"],
            "oov_auprc": se_oov_prc, "seen_auprc": se_seen_prc
        },
        "dual_channel_fusion": {
            "auroc": dual_res["auroc"], "auprc": dual_res["auprc"],
            "f1": dual_res["f1"], "precision": dual_res["precision"], "recall": dual_res["recall"],
            "oov_auprc": du_oov_prc, "seen_auprc": du_seen_prc
        }
    }
    with open(out_json, "w") as f:
        json.dump(results_payload, f, indent=2)
    print(f"\n[+] Saved complete ablation results payload to: {out_json}")

if __name__ == "__main__":
    main()
