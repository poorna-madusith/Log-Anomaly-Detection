"""
Master Drift Robustness Benchmark (0%, 10%, 20%, 30% Drift Tiers)
========================================================================
Directly evaluates architectural resilience against simulated CI/CD developer log drift:
- Tier 0%: Clean baseline control (0% mutation)
- Tier 10%: Minor incremental developer updates (10% mutation to <OOV>)
- Tier 20%: Moderate API refactoring (20% mutation to <OOV>)
- Tier 30%: Major system restructuring (30% mutation to <OOV>)

Models Evaluated:
1. Structural-Only Baseline (DeepLog equivalent)
2. Semantic-Only Baseline (Ablation model)
3. Dual-Channel Fusion Architecture (Proposed)

Convention:
- ID 0: <PAD> (strictly preserved, never mutated)
- ID 1: <OOV> (injected drift target token)
- Thresholds: Operating thresholds frozen strictly on validation data (no test snooping)
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from datetime import datetime, timezone
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score, precision_recall_curve

# Add parent directory to access shared components
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, parent_dir)
from models.model import DualChannelBiLSTM
from models.baseline_models import StructuralBaselineBiLSTM, SemanticBaselineBiLSTM
from models.dataset import create_dataloader

def parse_args():
    p = argparse.ArgumentParser(description="Run 0%-30% Drift Robustness Benchmark")
    p.add_argument("--dataset", type=str, default="HDFS", choices=["BGL", "HDFS", "Thunderbird"],
                   help="Dataset to benchmark on (proposal default: HDFS)")
    p.add_argument("--batch_size", type=int, default=256, help="Evaluation batch size")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return p.parse_args()

def calibrate_val_threshold(model, val_loader, device, is_semantic=False, is_dual=False):
    model.eval()
    val_y, val_probs = [], []
    with torch.no_grad():
        for struct_seq, sem_seq, y in val_loader:
            struct_seq = struct_seq.to(device)
            sem_seq = sem_seq.to(device)
            if is_dual:
                logits, _ = model(struct_seq, sem_seq)
            elif is_semantic:
                logits, _ = model(struct_seq, sem_seq)
            else:
                logits, _ = model(struct_seq)
            probs = torch.sigmoid(logits)
            val_probs.extend(probs.cpu().numpy())
            val_y.extend(y.numpy())
            
    val_y = np.array(val_y)
    val_probs = np.array(val_probs)
    prec_v, rec_v, thresh_v = precision_recall_curve(val_y, val_probs)
    f1_v = 2 * (prec_v * rec_v) / (prec_v + rec_v + 1e-9)
    best_idx = np.argmax(f1_v)
    tau = float(thresh_v[best_idx]) if best_idx < len(thresh_v) else 0.5
    return tau

def evaluate_drift_tier(model, test_loader, tau, drift_ratio, mutation_seed, device, is_semantic=False, is_dual=False):
    model.eval()
    all_y, all_probs = [], []
    # Each model gets a fresh local generator with the same seed. Because the
    # loader is ordered and batch shapes are identical, each model receives the
    # exact same Bernoulli mutation mask without materializing the full test set.
    mutation_generator = torch.Generator(device="cpu")
    mutation_generator.manual_seed(mutation_seed)
    
    with torch.no_grad():
        for struct_seq, sem_seq, y in test_loader:
            # Inject structural drift (mutating valid templates to ID 1 <OOV>)
            # Note: Semantic vectors remain untouched, simulating preserved semantics
            if drift_ratio > 0.0:
                random_prob = torch.rand(struct_seq.shape, generator=mutation_generator)
                drift_mask = (random_prob < drift_ratio) & (struct_seq > 1)
                struct_seq = struct_seq.clone()
                struct_seq[drift_mask] = 1
                
            struct_seq = struct_seq.to(device)
            sem_seq = sem_seq.to(device)
            
            if is_dual:
                logits, _ = model(struct_seq, sem_seq)
            elif is_semantic:
                logits, _ = model(struct_seq, sem_seq)
            else:
                logits, _ = model(struct_seq)
                
            probs = torch.sigmoid(logits)
            all_probs.extend(probs.cpu().numpy())
            all_y.extend(y.numpy())
            
    y_true = np.array(all_y)
    y_prob = np.array(all_probs)
    y_pred = (y_prob >= tau).astype(int)
    
    auroc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) >= 2 else float("nan")
    auprc = float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) >= 2 else float("nan")
    pos_f1 = float(f1_score(y_true, y_pred, zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    
    return {
        "auroc": auroc,
        "auprc": auprc,
        "pos_f1": pos_f1,
        "macro_f1": macro_f1,
        "precision": prec,
        "recall": rec
    }

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    base_dir = parent_dir
    data_dir = os.path.join(base_dir, "artifacts", args.dataset, "data")
    save_dir = os.path.join(base_dir, "artifacts", args.dataset, "checkpoints")
    results_dir = os.path.join(base_dir, "artifacts", args.dataset, "drift")
    os.makedirs(results_dir, exist_ok=True)
    
    print("\n" + "#"*75)
    print(f" CONTROLLED DRIFT ROBUSTNESS BENCHMARK (0%, 10%, 20%, 30%) - [{args.dataset}]")
    print(f" Device: {device} | Batch Size: {args.batch_size} | Seed: {args.seed}")
    print("#"*75)
    
    # 1. Load Data
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
    
    # 2. Load Checkpoints
    print("\n[*] Loading Trained Model Checkpoints...")
    dual_ckpt = os.path.join(save_dir, "best_model.pth")
    struct_ckpt = os.path.join(save_dir, "baseline_structural.pth")
    sem_ckpt = os.path.join(save_dir, "baseline_semantic.pth")
    
    dual_state = torch.load(dual_ckpt, map_location=device, weights_only=True)
    vocab_size = dual_state["struct_embedding.weight"].shape[0]
    
    # Dual-Channel
    dual_model = DualChannelBiLSTM(vocab_size=vocab_size).to(device)
    dual_model.load_state_dict(dual_state)
    
    # Structural Baseline
    struct_model = StructuralBaselineBiLSTM(vocab_size=vocab_size).to(device)
    struct_model.load_state_dict(torch.load(struct_ckpt, map_location=device, weights_only=True))
    
    # Semantic Baseline
    sem_model = SemanticBaselineBiLSTM(sem_input_dim=384).to(device)
    sem_model.load_state_dict(torch.load(sem_ckpt, map_location=device, weights_only=True))
    
    # 3. Calibrate Operating Thresholds Strictly on Validation Data
    print("\n[*] Calibrating Operating Thresholds (tau*) on Validation Split...")
    tau_dual = calibrate_val_threshold(dual_model, val_loader, device, is_dual=True)
    tau_struct = calibrate_val_threshold(struct_model, val_loader, device, is_semantic=False)
    tau_sem = calibrate_val_threshold(sem_model, val_loader, device, is_semantic=True)
    
    print(f"  [+] Dual-Channel Threshold:    tau* = {tau_dual:.4f}")
    print(f"  [+] Structural Baseline tau*:  tau* = {tau_struct:.4f}")
    print(f"  [+] Semantic Baseline tau*:    tau* = {tau_sem:.4f}")
    
    # 4. Run Benchmark across Drift Tiers
    drift_tiers = [
        ("0%", 0.0),
        ("10%", 0.10),
        ("20%", 0.20),
        ("30%", 0.30)
    ]
    
    benchmark_results = []
    
    print("\n" + "="*85)
    print(f"{'Tier':<6} | {'Model Architecture':<30} | {'AUROC':<7} | {'AUPRC':<7} | {'Pos F1':<7} | {'Precision':<9} | {'Recall':<7}")
    print("="*85)
    
    for tier_index, (tier_name, ratio) in enumerate(drift_tiers):
        tier_seed = args.seed + tier_index
        st_res = evaluate_drift_tier(struct_model, test_loader, tau_struct, ratio, tier_seed, device, is_semantic=False)
        se_res = evaluate_drift_tier(sem_model, test_loader, tau_sem, ratio, tier_seed, device, is_semantic=True)
        du_res = evaluate_drift_tier(dual_model, test_loader, tau_dual, ratio, tier_seed, device, is_dual=True)
        
        print(f"{tier_name:<6} | {'1. Structural-Only (DeepLog)':<30} | {st_res['auroc']:.4f}  | {st_res['auprc']:.4f}  | {st_res['pos_f1']:.4f}  | {st_res['precision']:.4f}    | {st_res['recall']:.4f}")
        print(f"{'':<6} | {'2. Semantic-Only (Ablation)':<30} | {se_res['auroc']:.4f}  | {se_res['auprc']:.4f}  | {se_res['pos_f1']:.4f}  | {se_res['precision']:.4f}    | {se_res['recall']:.4f}")
        print(f"{'':<6} | {'3. Dual-Channel Fusion (Ours)':<30} | {du_res['auroc']:.4f}  | {du_res['auprc']:.4f}  | {du_res['pos_f1']:.4f}  | {du_res['precision']:.4f}    | {du_res['recall']:.4f}")
        print("-" * 85)
        
        benchmark_results.append({
            "drift_tier": tier_name,
            "drift_ratio": ratio,
            "mutation_seed": tier_seed,
            "structural_only": st_res,
            "semantic_only": se_res,
            "dual_channel": du_res
        })
        
    # 5. Save Results
    out_json = os.path.join(results_dir, "drift_benchmark_results.json")
    results_payload = {
        "dataset": args.dataset,
        "seed": args.seed,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mutation_protocol": "same seeded structural-token mutation mask per tier and test batch across all models",
        "results": benchmark_results,
    }
    with open(out_json, "w") as f:
        json.dump(results_payload, f, indent=2)
    print(f"\n[+] Saved complete drift benchmark results to: {out_json}")

if __name__ == "__main__":
    main()
