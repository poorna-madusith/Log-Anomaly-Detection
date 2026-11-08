"""
Pipeline Integrity and Scientific Sanity Verification
========================================================================
Runs 5 independent verification tests across HDFS, BGL, and Thunderbird:
1. Checkpoint Synchronization Check (Sampled model outputs vs. saved predictions)
2. Permutation / Null-Hypothesis Sanity Check (One seeded shuffle should be near chance)
3. Data Splitting Leakage Check (Disjoint source identities across Train, Val, and Test)
4. OOV Semantic-Input Ablation (Compares the same model with real vs. occluded semantics)
5. Threshold Calibration Diagnostic (Compares validation-frozen and test-oracle references)
"""

import os
import sys
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, f1_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
from models.model import DualChannelBiLSTM
from models.dataset import create_dataloader

def print_header(title):
    print("\n" + "="*75)
    print(f" {title}")
    print("="*75)

def test_1_prediction_synchronization(base_dir):
    print_header("TEST 1: CHECKPOINT-TO-PREDICTION SAMPLE SYNCHRONIZATION")
    print("Goal: Compare the first two test batches with saved outputs (tolerance 1e-4).")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = [
        ("HDFS", os.path.join(base_dir, "artifacts", "HDFS")),
        ("BGL", os.path.join(base_dir, "artifacts", "BGL")),
        ("Thunderbird", os.path.join(base_dir, "artifacts", "Thunderbird"))
    ]
    
    for name, path in datasets:
        model_path = os.path.join(path, "checkpoints", "best_model.pth")
        preds_path = os.path.join(path, "checkpoints", "test_predictions.npz")
        data_path = os.path.join(path, "data", "test_data.npz")
        mat_path = os.path.join(path, "data", "string_matrix.npy")
        
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        vocab_size = state_dict["struct_embedding.weight"].shape[0]
        model = DualChannelBiLSTM(vocab_size=vocab_size).to(device)
        model.load_state_dict(state_dict)
        model.eval()
        
        # Load first 2 batches of test data
        loader, _ = create_dataloader(data_path, mat_path, batch_size=256, shuffle=False)
        sample_probs = []
        with torch.no_grad():
            for i, (struct, sem, _) in enumerate(loader):
                if i >= 2: break
                logits, _ = model(struct.to(device), sem.to(device))
                sample_probs.extend(torch.sigmoid(logits).cpu().numpy())
                
        sample_probs = np.array(sample_probs)
        with np.load(preds_path) as f:
            saved_probs = f["probs"][:len(sample_probs)]
            
        max_diff = np.max(np.abs(sample_probs - saved_probs))
        status = "PASSED" if max_diff < 1e-4 else "FAILED"
        print(f"  [{name}] Max Probability Discrepancy: {max_diff:.2e} -> {status}")
        assert max_diff < 1e-4, f"{name} predictions do not match checkpoint weights!"

def test_2_permutation_sanity(base_dir):
    print_header("TEST 2: LABEL PERMUTATION NULL-HYPOTHESIS TEST")
    print("Goal: A single seeded label shuffle should produce a near-chance AUROC.")
    print("      This is a null sanity check, not proof that all reported results are valid.")
    
    datasets = ["HDFS", "BGL", "Thunderbird"]
    rng = np.random.default_rng(42)
    
    for name in datasets:
        preds_path = os.path.join(base_dir, "artifacts", name, "checkpoints", "test_predictions.npz")
        with np.load(preds_path) as f:
            y, probs = f["targets"], f["probs"]
            
        true_auroc = roc_auc_score(y, probs)
        true_auprc = average_precision_score(y, probs)
        prevalence = np.mean(y)
        
        # Scramble labels
        y_scrambled = rng.permutation(y)
        null_auroc = roc_auc_score(y_scrambled, probs)
        null_auprc = average_precision_score(y_scrambled, probs)
        
        print(f"\n  [{name}]")
        print(f"    Real Model Scores:     AUROC = {true_auroc:.4f} | AUPRC = {true_auprc:.4f}")
        print(f"    Scrambled Null Scores: AUROC = {null_auroc:.4f} (expect ~0.50) | AUPRC = {null_auprc:.4f} (expect ~{prevalence:.4f})")
        
        assert abs(null_auroc - 0.50) < 0.05, f"{name} failed null permutation test!"
        print(f"    --> PASSED: This shuffled-label sample is consistent with the null expectation.")

def test_3_split_isolation(base_dir):
    print_header("TEST 3: DATA SPLIT ISOLATION & LEAKAGE CHECK")
    print("Goal: Verify unique source identities do not overlap across Train, Val, and Test.")
    
    datasets = ["HDFS", "BGL", "Thunderbird"]
    for name in datasets:
        train_p = os.path.join(base_dir, "artifacts", name, "data", "train_data.npz")
        val_p = os.path.join(base_dir, "artifacts", name, "data", "val_data.npz")
        test_p = os.path.join(base_dir, "artifacts", name, "data", "test_data.npz")
        
        with np.load(train_p) as tr, np.load(val_p) as va, np.load(test_p) as te:
            n_tr, n_va, n_te = len(tr["y"]), len(va["y"]), len(te["y"])
            anom_tr, anom_va, anom_te = int(tr["y"].sum()), int(va["y"].sum()), int(te["y"].sum())

            if name == "Thunderbird":
                required = ("chunk_ids", "chunk_window_indices")
                for split_name, split in (("train", tr), ("val", va), ("test", te)):
                    missing = [key for key in required if key not in split]
                    if missing:
                        raise AssertionError(
                            f"Thunderbird {split_name} data lacks {missing}; rerun preprocessing to save source identities."
                        )
                split_keys = [
                    list(zip(split["chunk_ids"].tolist(), split["chunk_window_indices"].tolist()))
                    for split in (tr, va, te)
                ]
                for split_name, keys in zip(("train", "val", "test"), split_keys):
                    if len(set(keys)) != len(keys):
                        raise AssertionError(f"{name} {split_name} contains duplicate source chunk/window identities.")
                key_sets = [set(keys) for keys in split_keys]
                if any(keys != sorted(keys) for keys in split_keys):
                    raise AssertionError("Thunderbird source windows are not in chronological order within a split.")
                if not (
                    max(split_keys[0]) < min(split_keys[1])
                    and max(split_keys[1]) < min(split_keys[2])
                ):
                    raise AssertionError("Thunderbird source windows are not chronologically ordered across splits.")
            else:
                if any("source_ids" not in split for split in (tr, va, te)):
                    raise AssertionError(f"{name} data lacks source_ids; rerun preprocessing to save source identities.")
                split_ids = [split["source_ids"].tolist() for split in (tr, va, te)]
                for split_name, ids in zip(("train", "val", "test"), split_ids):
                    if len(set(ids)) != len(ids):
                        raise AssertionError(f"{name} {split_name} contains duplicate source identities.")
                key_sets = [set(ids) for ids in split_ids]
                if name == "BGL":
                    source_ids = [np.asarray(split["source_ids"]) for split in (tr, va, te)]
                    if not (source_ids[0].max() < source_ids[1].min() and source_ids[1].max() < source_ids[2].min()):
                        raise AssertionError("BGL source window IDs are not strictly chronological across splits.")

            if any(key_sets[i] & key_sets[j] for i, j in ((0, 1), (0, 2), (1, 2))):
                raise AssertionError(f"{name} source identities overlap across train, validation, and test.")

        total = n_tr + n_va + n_te
        print(f"  [{name}] Split Partitioning (Total: {total:,} windows):")
        print(f"    Train: {n_tr:,} ({n_tr/total*100:.1f}%) | Anomalies: {anom_tr:,} ({anom_tr/n_tr*100:.2f}%)")
        print(f"    Val:   {n_va:,} ({n_va/total*100:.1f}%) | Anomalies: {anom_va:,} ({anom_va/n_va*100:.2f}%)")
        print(f"    Test:  {n_te:,} ({n_te/total*100:.1f}%) | Anomalies: {anom_te:,} ({anom_te/n_te*100:.2f}%)")
        print("    --> PASSED: Source identities are unique and disjoint across splits.")

def test_4_oov_semantic_input_ablation(base_dir):
    print_header("TEST 4: OOV SEMANTIC-INPUT ABLATION")
    print("Goal: Compare the dual model with real and occluded semantic inputs on identical OOV windows.")

    data_dir = os.path.join(base_dir, "artifacts", "BGL", "data")
    data_path = os.path.join(data_dir, "test_data.npz")
    model_path = os.path.join(base_dir, "artifacts", "BGL", "checkpoints", "best_model.pth")
    with np.load(data_path) as data:
        X_struct, targets = data["X_struct"], data["y"]

    is_oov = (X_struct == 1).any(axis=1)
    if not is_oov.any() or len(np.unique(targets[is_oov])) < 2:
        raise AssertionError("BGL OOV windows must contain both classes for semantic-input ablation.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    vocab_size = state_dict["struct_embedding.weight"].shape[0]
    model = DualChannelBiLSTM(vocab_size=vocab_size).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    loader, _ = create_dataloader(
        data_path, os.path.join(data_dir, "string_matrix.npy"), batch_size=256, shuffle=False
    )
    with_semantics, without_semantics = [], []
    with torch.no_grad():
        for struct_seq, sem_seq, _ in loader:
            struct_seq, sem_seq = struct_seq.to(device), sem_seq.to(device)
            logits, _ = model(struct_seq, sem_seq)
            masked_logits, _ = model(struct_seq, torch.zeros_like(sem_seq))
            with_semantics.extend(torch.sigmoid(logits).cpu().numpy())
            without_semantics.extend(torch.sigmoid(masked_logits).cpu().numpy())

    with_semantics = np.asarray(with_semantics)
    without_semantics = np.asarray(without_semantics)
    if len(with_semantics) != len(targets):
        raise AssertionError("BGL predictions and source labels are misaligned.")

    y_oov = targets[is_oov]
    auc_with = roc_auc_score(y_oov, with_semantics[is_oov])
    ap_with = average_precision_score(y_oov, with_semantics[is_oov])
    auc_without = roc_auc_score(y_oov, without_semantics[is_oov])
    ap_without = average_precision_score(y_oov, without_semantics[is_oov])
    print(f"  OOV windows: {int(is_oov.sum()):,}")
    print(f"  With semantic input:     AUROC = {auc_with:.4f} | AP = {ap_with:.4f}")
    print(f"  Semantic input occluded: AUROC = {auc_without:.4f} | AP = {ap_without:.4f}")
    print(f"  AP change from semantic input: {ap_with - ap_without:+.4f}")
    if ap_with <= ap_without:
        raise AssertionError(
            "Semantic-input occlusion did not reduce OOV average precision; this checkpoint shows no OOV performance gain from semantic input under this ablation."
        )
    print("  --> PASSED: Semantic input contributes to OOV performance; this ablation is not causal proof.")

def test_5_threshold_calibration_diagnostic(base_dir):
    print_header("TEST 5: THRESHOLD CALIBRATION DIAGNOSTIC (VAL vs. TEST ORACLE)")
    print("Goal: Recompute a validation-frozen threshold and compare with a test-only oracle reference.")
    print("      This diagnostic does not prove which split the training/evaluation code used.")
    
    datasets = ["HDFS", "BGL", "Thunderbird"]
    for name in datasets:
        val_preds_path = os.path.join(base_dir, "artifacts", name, "checkpoints", "val_predictions.npz")
        test_preds_path = os.path.join(base_dir, "artifacts", name, "checkpoints", "test_predictions.npz")
        
        with np.load(val_preds_path) as v, np.load(test_preds_path) as t:
            vy, vp = v["targets"], v["probs"]
            ty, tp = t["targets"], t["probs"]
            
        # Calibrate on validation
        pv, rv, tv = precision_recall_curve(vy, vp)
        f1v = 2 * (pv[:-1] * rv[:-1]) / (pv[:-1] + rv[:-1] + 1e-9)
        best_v = int(np.argmax(f1v)) if len(tv) else -1
        tau_val = float(tv[best_v]) if best_v >= 0 else 0.5
        
        # Test performance at frozen tau
        test_pred_val = (tp >= tau_val).astype(int)
        f1_calibrated = f1_score(ty, test_pred_val, zero_division=0)
        
        # Oracle on test (cheating reference)
        pt, rt, tt = precision_recall_curve(ty, tp)
        f1t = 2 * (pt[:-1] * rt[:-1]) / (pt[:-1] + rt[:-1] + 1e-9)
        best_t = int(np.argmax(f1t)) if len(tt) else -1
        tau_oracle = float(tt[best_t]) if best_t >= 0 else 0.5
        f1_oracle = float(f1t[best_t]) if best_t >= 0 else float(f1_score(ty, tp >= 0.5, zero_division=0))
        
        print(f"  [{name}]")
        print(f"    Validation-Frozen Threshold tau* = {tau_val:.4f} -> Test F1 = {f1_calibrated:.4f}")
        print(f"    Cheating Oracle Threshold    tau  = {tau_oracle:.4f} -> Test F1 = {f1_oracle:.4f}")
        print(f"    Difference (Generalization Gap): {abs(f1_oracle - f1_calibrated):.4f}")
        print("    --> Diagnostic only: the oracle uses test labels and must not be used as the operating threshold.")

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print("\n" + "#"*75)
    print(" SCIENTIFIC INTEGRITY & SOUNDNESS SUITE (5-POINT AUDIT)")
    print("#"*75)
    
    test_1_prediction_synchronization(base_dir)
    test_2_permutation_sanity(base_dir)
    test_3_split_isolation(base_dir)
    test_4_oov_semantic_input_ablation(base_dir)
    test_5_threshold_calibration_diagnostic(base_dir)
    
    print_header("ASSERTED INTEGRITY CHECKS PASSED; THRESHOLD COMPARISON IS DIAGNOSTIC")
    print("Conclusions:")
    print(" 1. Sampled predictions match checkpoint outputs within the configured tolerance.")
    print(" 2. One seeded shuffled-label sample is consistent with the near-chance null expectation.")
    print(" 3. Saved source identities are unique and disjoint across splits.")
    print(" 4. Semantic-input occlusion checks whether OOV average precision falls when semantic inputs are zeroed.")
    print(" 5. Validation and test-oracle thresholds are reported for comparison; this script does not verify the evaluator code path.")
    print("="*75 + "\n")

if __name__ == "__main__":
    main()
