"""
HDFS Training Script (FYP Final Implementation)
========================================================================
Trains the Dual-Channel BiLSTM model on the HDFS dataset.
Key Academic Features:
1. Dynamic Positional Weighting for class imbalance.
2. Checkpoints based on Validation AUPRC.
3. Saves raw test probabilities for block-bootstrap evaluation.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from models.dataset import create_dataloader
from models.model import DualChannelBiLSTM

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="./artifacts/HDFS/data")
    p.add_argument("--save_dir", type=str, default="./artifacts/HDFS/checkpoints")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def evaluate(model, loader, device):
    model.eval()
    all_y, all_p = [], []
    with torch.no_grad():
        for struct_seq, sem_seq, y in loader:
            struct_seq, sem_seq = struct_seq.to(device), sem_seq.to(device)
            logits, _ = model(struct_seq, sem_seq)
            probs = torch.sigmoid(logits)
            
            all_y.extend(y.cpu().numpy())
            all_p.extend(probs.cpu().numpy())
            
    auroc = roc_auc_score(all_y, all_p)
    auprc = average_precision_score(all_y, all_p)
    return auroc, auprc, np.array(all_y), np.array(all_p)

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("\n[*] Loading HDFS datasets...")
    train_loader, train_ds = create_dataloader(os.path.join(args.data_dir, "train_data.npz"), os.path.join(args.data_dir, "string_matrix.npy"), args.batch_size, shuffle=True)
    val_loader, _ = create_dataloader(os.path.join(args.data_dir, "val_data.npz"), os.path.join(args.data_dir, "string_matrix.npy"), args.batch_size, shuffle=False)
    test_loader, _ = create_dataloader(os.path.join(args.data_dir, "test_data.npz"), os.path.join(args.data_dir, "string_matrix.npy"), args.batch_size, shuffle=False)

    print(f"[*] Initializing DualChannelBiLSTM on {device}")
    model = DualChannelBiLSTM(vocab_size=5000).to(device)
    
    pos_weight = train_ds.get_pos_weight().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_auprc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for struct_seq, sem_seq, y in pbar:
            optimizer.zero_grad()
            logits, _ = model(struct_seq.to(device), sem_seq.to(device))
            loss = criterion(logits, y.to(device))
            loss.backward()
            optimizer.step()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        val_auroc, val_auprc, _, _ = evaluate(model, val_loader, device)
        print(f"  --> Val AUROC: {val_auroc:.4f} | Val AUPRC: {val_auprc:.4f}")
        
        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best_model.pth"))

    print("\n[*] Evaluating Best Model on Validation Set...")
    model.load_state_dict(torch.load(os.path.join(args.save_dir, "best_model.pth"), weights_only=True))
    val_auroc, val_auprc, val_y, val_p = evaluate(model, val_loader, device)
    val_preds_path = os.path.join(args.save_dir, "val_predictions.npz")
    np.savez_compressed(val_preds_path, targets=val_y, probs=val_p)
    print(f"[+] Saved validation predictions to {val_preds_path}")

    print("\n[*] Evaluating Best Model on Test Set...")
    test_auroc, test_auprc, test_y, test_p = evaluate(model, test_loader, device)
    
    print(f"==================================================")
    print(f" FINAL TEST SCORES: AUROC = {test_auroc:.4f} | AUPRC = {test_auprc:.4f}")
    print(f"==================================================")
    
    preds_path = os.path.join(args.save_dir, "test_predictions.npz")
    np.savez_compressed(preds_path, targets=test_y, probs=test_p)
    print(f"[+] Saved raw predictions to {preds_path}")

if __name__ == "__main__":
    main()
