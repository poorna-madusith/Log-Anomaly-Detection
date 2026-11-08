"""
HDFS Preprocessing Script (FYP Final Implementation)
========================================================================
Extracts and windows HDFS block sessions.
Bug Fixes & Methodological Upgrades:
1. True Padding Tokens: Structural ID 0 and Semantic Index 0 are strictly 
   reserved for <PAD> (mapped to an all-zero vector). Real messages start at Index 1.
2. Unseen Drain templates are assigned ID 1 (<OOV>). Drain clusters start at 2.
3. Default output directory fixed to ./HDFS/data to align with train_hdfs.py.
"""

import os
import re
import sys
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from models.semantic_cleaner import clean_line_for_semantics, encode_string_vocabulary

def parse_args():
    p = argparse.ArgumentParser(description="Preprocess HDFS logs")
    p.add_argument("--log_file", type=str, default=r"../../datasets/HDFS/HDFS.log")
    p.add_argument("--label_file", type=str, default=r"../../datasets/HDFS/preprocessed/anomaly_label.csv")
    p.add_argument("--output_dir", type=str, default="./artifacts/HDFS/data")
    p.add_argument("--window_size", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "="*72)
    print("HDFS PREPROCESSING (Block-level with Strict <PAD> Reservation)")
    print("="*72)
    
    # 1. Load Ground Truth Labels
    print(f"[*] Loading labels from {args.label_file}...")
    label_df = pd.read_csv(args.label_file)
    anomaly_map = {row['BlockId']: 1 if row['Label'] == 'Anomaly' else 0 for _, row in label_df.iterrows()}
    
    # 2. Group Log Lines by BlockId
    print(f"[*] Grouping logs by BlockId from {args.log_file}...")
    blocks = {}
    blk_pattern = re.compile(r'(blk_-?\d+)')
    
    with open(args.log_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in tqdm(f, desc="Reading lines", ncols=80):
            match = blk_pattern.search(line)
            if match:
                blk_id = match.group(1)
                if blk_id not in blocks:
                    blocks[blk_id] = []
                blocks[blk_id].append(line.strip())
                
    valid_blocks = {k: v for k, v in blocks.items() if k in anomaly_map}
    print(f"[+] Extracted {len(valid_blocks):,} labeled blocks.")
    
    # 3. Splits (70% Train, 10% Val, 20% Test)
    block_keys = list(valid_blocks.keys())
    n_total = len(block_keys)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.10)
    
    train_keys = block_keys[:n_train]
    val_keys = block_keys[n_train:n_train + n_val]
    test_keys = block_keys[n_train + n_val:]
    
    # 4. Fit Drain3 ONLY on Train Blocks
    print("\n[*] Fitting Drain3 on Training Blocks...")
    config = TemplateMinerConfig()
    config.drain_depth = 4
    config.profiling_enabled = False
    miner = TemplateMiner(config=config)
    
    for k in tqdm(train_keys, desc="Fitting Drain", ncols=80):
        for line in valid_blocks[k]:
            miner.add_log_message(line)
            
    print(f"[+] Drain discovered {len(miner.drain.clusters)} templates.")
    
    # 5. Build Tensors with True <PAD> Reservation (Index 0)
    print("\n[*] Building tensors and semantic vocabulary...")
    # Index 0 is strictly reserved for <PAD>
    string_to_idx = {"<PAD>": 0}
    vocab_strings = ["<PAD>"]
    
    def get_or_add_string(cleaned: str):
        if cleaned not in string_to_idx:
            string_to_idx[cleaned] = len(vocab_strings)
            vocab_strings.append(cleaned)
        return string_to_idx[cleaned]

    def build_split(keys, name):
        N = len(keys)
        # Initialized with 0, which corresponds to <PAD> for both channels
        X_struct = np.zeros((N, args.window_size), dtype=np.int64)
        X_sem_idx = np.zeros((N, args.window_size), dtype=np.int64)
        y = np.zeros(N, dtype=np.float32)
        
        for i, k in enumerate(tqdm(keys, desc=f"Processing {name}", ncols=80)):
            lines = valid_blocks[k]
            y[i] = anomaly_map[k]
            
            # Truncate to window size if needed
            seq_len = min(len(lines), args.window_size)
            for t in range(seq_len):
                line = lines[t]
                match = miner.match(line)
                # ID 0 = PAD, ID 1 = OOV, Templates start at 2
                X_struct[i, t] = (match.cluster_id + 1) if match else 1
                
                cleaned = clean_line_for_semantics(line)
                X_sem_idx[i, t] = get_or_add_string(cleaned)
                
        out_file = os.path.join(args.output_dir, f"{name}_data.npz")
        np.savez_compressed(
            out_file,
            X_struct=X_struct,
            X_sem_idx=X_sem_idx,
            y=y,
            source_ids=np.asarray(keys, dtype=np.str_),
        )
        print(f"  Saved {name} (N={N:,}, Anom={int(sum(y)):,}) -> {out_file}")

    build_split(train_keys, "train")
    build_split(val_keys, "val")
    build_split(test_keys, "test")
    
    # Save fitted Drain parser
    import pickle
    miner_path = os.path.join(args.output_dir, "drain_miner.pkl")
    with open(miner_path, "wb") as f:
        pickle.dump(miner, f)
    print(f"[+] Saved fitted Drain3 parser: {miner_path}")
    
    # 6. Encode Vocabulary (Setting Row 0 to an all-zero vector for <PAD>)
    print("\n[*] Encoding shared string vocabulary...")
    string_matrix = encode_string_vocabulary(vocab_strings)
    # Guarantee that row 0 (<PAD>) is mathematically an all-zero vector
    string_matrix[0] = 0.0
    
    out_matrix = os.path.join(args.output_dir, "string_matrix.npy")
    np.save(out_matrix, string_matrix)
    print(f"[+] Saved string matrix: {out_matrix} (Shape: {string_matrix.shape})")

if __name__ == "__main__":
    main()
