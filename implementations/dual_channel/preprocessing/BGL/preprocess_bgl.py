"""
BGL Preprocessing Script (FYP Final Implementation)
========================================================================
This script processes the raw BGL supercomputer log into windowed sequences.
Key Academic Features:
1. Strict Chronological Splitting (70/10/20) to prevent data leakage and 
   accurately simulate real-world log drift.
2. Unified Semantic Normalization: Strips dynamic variables (IPs, node IDs)
   to ensure the SentenceTransformer focuses on structural semantics.
3. Drain Template Extraction: Extracts structural IDs for the dual-channel model.
"""

import os
import sys
import numpy as np
import argparse
from tqdm import tqdm
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

# Import shared core components
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from models.semantic_cleaner import clean_line_for_semantics, encode_string_vocabulary

def parse_args():
    p = argparse.ArgumentParser(description="Preprocess BGL logs for Dual-Channel Model")
    p.add_argument("--log_file", type=str, default=r"../../datasets/BGL/BGL.log", help="Path to raw BGL.log")
    p.add_argument("--output_dir", type=str, default="./artifacts/BGL/data", help="Output directory for NPZ files")
    p.add_argument("--window_size", type=int, default=50, help="Number of log lines per sequence window")
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # ---------------------------------------------------------
    # 1. Read and Extract Raw Lines
    # ---------------------------------------------------------
    print("\n[*] Reading BGL lines...")
    raw_lines, labels = [], []
    with open(args.log_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in tqdm(f, desc="Reading raw file"):
            parts = line.strip().split()
            if len(parts) < 10: 
                continue
            # BGL labels are in the first column: "-" means normal, otherwise anomaly
            labels.append(0 if parts[0] == "-" else 1)
            # The actual log message starts at index 9
            raw_lines.append(" ".join(parts[9:]))
            
    n_windows = len(raw_lines) // args.window_size
    n_train = int(n_windows * 0.70)
    n_val = int(n_windows * 0.10)
    
    # ---------------------------------------------------------
    # 2. Fit Drain Parser (ONLY on Training Data to prevent leakage)
    # ---------------------------------------------------------
    print("\n[*] Fitting Drain3 Parser on Training windows...")
    config = TemplateMinerConfig()
    config.drain_depth = 4
    config.profiling_enabled = False
    miner = TemplateMiner(config=config)
    
    train_lines = raw_lines[:n_train * args.window_size]
    for line in tqdm(train_lines, desc="Drain fit"):
        miner.add_log_message(line)
        
    print(f"[+] Drain discovered {len(miner.drain.clusters)} templates.")
        
    # ---------------------------------------------------------
    # 3. Process Windows (Extract Structural & Semantic IDs)
    # ---------------------------------------------------------
    # Index 0 is strictly reserved for <PAD>
    string_to_idx = {"<PAD>": 0}
    vocab_strings = ["<PAD>"]
    
    def get_or_add(cleaned: str):
        """Maintains a unique vocabulary of cleaned strings."""
        if cleaned not in string_to_idx:
            string_to_idx[cleaned] = len(vocab_strings)
            vocab_strings.append(cleaned)
        return string_to_idx[cleaned]

    def process_split(start_idx, end_idx, name):
        """Processes a chronological slice of windows into NPZ tensors."""
        N = end_idx - start_idx
        # Default 0 is <PAD>
        X_struct = np.zeros((N, args.window_size), dtype=np.int64)
        X_sem_idx = np.zeros((N, args.window_size), dtype=np.int64)
        y = np.zeros(N, dtype=np.float32)
        
        for i in tqdm(range(N), desc=f"Processing {name} split"):
            w_start = (start_idx + i) * args.window_size
            w_end = w_start + args.window_size
            
            w_lines = raw_lines[w_start:w_end]
            w_labels = labels[w_start:w_end]
            
            # Label window as 1 if ANY line in the window is anomalous
            y[i] = 1 if sum(w_labels) > 0 else 0
            
            for t, line in enumerate(w_lines):
                # Structural feature extraction:
                # ID 0 = <PAD>, ID 1 = <OOV> (unmatched), Templates start at 2
                match = miner.match(line)
                X_struct[i, t] = (match.cluster_id + 1) if match else 1
                
                # Semantic feature extraction
                cleaned = clean_line_for_semantics(line)
                X_sem_idx[i, t] = get_or_add(cleaned)
                
        out_path = os.path.join(args.output_dir, f"{name}_data.npz")
        np.savez_compressed(
            out_path,
            X_struct=X_struct,
            X_sem_idx=X_sem_idx,
            y=y,
            source_ids=np.arange(start_idx, end_idx, dtype=np.int64),
        )
        print(f"  [+] Saved {name}: {N:,} windows, {int(sum(y)):,} anomalies")

    print("\n[*] Processing chronological splits...")
    process_split(0, n_train, "train")
    process_split(n_train, n_train + n_val, "val")
    process_split(n_train + n_val, n_windows, "test")
    
    # Save fitted Drain parser
    import pickle
    miner_path = os.path.join(args.output_dir, "drain_miner.pkl")
    with open(miner_path, "wb") as f:
        pickle.dump(miner, f)
    print(f"[+] Saved fitted Drain3 parser: {miner_path}")
    
    # ---------------------------------------------------------
    # 4. Encode Semantic Vocabulary
    # ---------------------------------------------------------
    print("\n[*] Encoding semantic vocabulary with SentenceTransformer...")
    string_matrix = encode_string_vocabulary(vocab_strings)
    # Guarantee row 0 (<PAD>) is mathematically an all-zero vector
    string_matrix[0] = 0.0
    matrix_path = os.path.join(args.output_dir, "string_matrix.npy")
    np.save(matrix_path, string_matrix)
    print(f"[+] Saved string matrix: {matrix_path} (Shape: {string_matrix.shape})")

if __name__ == "__main__":
    main()
