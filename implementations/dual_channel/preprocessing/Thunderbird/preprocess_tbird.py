"""
Thunderbird Preprocessing Script (FYP Final Implementation)
========================================================================
This script processes the massive 30GB Thunderbird log.
Key Academic Features:
1. Chunked Sampling: Thunderbird suffers from a massive contiguous anomaly storm.
   To ensure representative train/val/test splits, this script takes 20 evenly 
   spaced chunks across the entire file, completely avoiding temporal bias.
2. Unified Semantic Normalization: Ensures the vocabulary matches BGL for 
   future zero-shot transfer learning.
"""

import os
import sys
import numpy as np
import argparse
from tqdm import tqdm
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from models.semantic_cleaner import clean_line_for_semantics, encode_string_vocabulary

def parse_args():
    p = argparse.ArgumentParser(description="Preprocess Thunderbird with Chunked Sampling")
    p.add_argument("--log_file", type=str, default=r"../../datasets/Thunderbird/Thunderbird.log")
    p.add_argument("--output_dir", type=str, default="./artifacts/Thunderbird/data")
    p.add_argument("--window_size", type=int, default=50)
    p.add_argument("--total_windows", type=int, default=100000, help="Total windows to extract across chunks")
    p.add_argument("--num_chunks", type=int, default=20, help="Number of chunks to prevent anomaly storm bias")
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # ---------------------------------------------------------
    # 1. Chunked Extraction Strategy
    # ---------------------------------------------------------
    file_size = os.path.getsize(args.log_file)
    # Space chunks evenly across the first 95% of the file
    chunk_offsets = np.linspace(0, file_size * 0.95, args.num_chunks, dtype=np.int64)
    windows_per_chunk = args.total_windows // args.num_chunks
    
    print(f"\n[*] Extracting {args.num_chunks} chunks evenly from 30GB Thunderbird log...")
    raw_lines, labels = [], []
    chunk_ids, chunk_window_indices = [], []
    
    with open(args.log_file, "r", encoding="utf-8", errors="ignore") as f:
        for chunk_id, offset in enumerate(tqdm(chunk_offsets, desc="Processing Chunks")):
            f.seek(offset)
            f.readline() # Discard partial line caused by random seek
            
            chunk_lines, chunk_labels = [], []
            target_lines = windows_per_chunk * args.window_size
            while len(chunk_lines) < target_lines:
                line = f.readline()
                if not line: break
                parts = line.strip().split()
                if len(parts) < 8: continue
                
                # Thunderbird labels: "-" is normal, anything else is anomalous
                chunk_labels.append(0 if parts[0] == "-" else 1)
                chunk_lines.append(" ".join(parts[7:]))

            # Discard an incomplete tail at each sampling boundary so windows
            # never combine events from two distant byte offsets.
            complete_lines = (len(chunk_lines) // args.window_size) * args.window_size
            complete_windows = complete_lines // args.window_size
            raw_lines.extend(chunk_lines[:complete_lines])
            labels.extend(chunk_labels[:complete_lines])
            chunk_ids.extend([chunk_id] * complete_windows)
            chunk_window_indices.extend(range(complete_windows))

    n_windows = len(raw_lines) // args.window_size
    if len(chunk_ids) != n_windows or len(chunk_window_indices) != n_windows:
        raise RuntimeError("Thunderbird source chunk metadata is not aligned with extracted windows.")
    n_train = int(n_windows * 0.70)
    n_val = int(n_windows * 0.10)
    
    # ---------------------------------------------------------
    # 2. Fit Drain Parser (ONLY on Training Data)
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
        if cleaned not in string_to_idx:
            string_to_idx[cleaned] = len(vocab_strings)
            vocab_strings.append(cleaned)
        return string_to_idx[cleaned]

    def process_split(start_idx, end_idx, name):
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
            y[i] = 1 if sum(w_labels) > 0 else 0
            
            for t, line in enumerate(w_lines):
                # Structural feature extraction:
                # ID 0 = <PAD>, ID 1 = <OOV> (unmatched), Templates start at 2
                match = miner.match(line)
                X_struct[i, t] = (match.cluster_id + 1) if match else 1
                X_sem_idx[i, t] = get_or_add(clean_line_for_semantics(line))
                
        out_path = os.path.join(args.output_dir, f"{name}_data.npz")
        np.savez_compressed(
            out_path,
            X_struct=X_struct,
            X_sem_idx=X_sem_idx,
            y=y,
            chunk_ids=np.asarray(chunk_ids[start_idx:end_idx], dtype=np.int64),
            chunk_window_indices=np.asarray(
                chunk_window_indices[start_idx:end_idx], dtype=np.int64
            ),
        )
        print(f"  [+] Saved {name}: {N:,} windows, {int(sum(y)):,} anomalies")

    print("\n[*] Processing chronological chunks into splits...")
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
