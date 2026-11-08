"""
Unified Text Cleaner & Semantic Encoder
========================================================================
Best Practice: A single source of truth for text normalization across ALL 
datasets (HDFS, BGL, Thunderbird) to ensure the semantic vocabulary is 
compatible for cross-system transfer learning.
"""

import re
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

def clean_line_for_semantics(raw_text: str) -> str:
    """
    Canonical normalization used by every dataset in the Final Pipeline.
    
    Order of operations:
    1. Strip Drain wildcard tokens <*>
    2. Mask IP addresses (e.g., 192.168.1.1, 10.0.0.1:8080)
    3. Mask hex literals (e.g., 0x1a2b)
    4. Mask BGL node IDs (e.g., R07-M1-N0-C:J00)
    5. Mask core IDs (e.g., core.3)
    6. Mask HDFS block IDs (e.g., blk_-1608999687919862906)
    7. Strip remaining standalone numbers (word-boundary)
    8. Collapse whitespace and lowercase
    """
    s = str(raw_text).strip()

    # 1. Drain wildcard
    s = re.sub(r'<\*>', ' ', s)
    # 2. IP addresses (IPv4 with optional port)
    s = re.sub(r'(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?', ' ', s)
    # 3. Hex literals (support both 0x and 0X)
    s = re.sub(r'0[xX][a-fA-F0-9]+', ' ', s)
    # 4. BGL node IDs (supporting hex cards e.g. NA..NF)
    s = re.sub(r'R\d{2}-M[0-9A-Za-z]-N[0-9A-Za-z]-[A-Za-z0-9\-:]+', ' ', s)
    # 5. Core IDs
    s = re.sub(r'core\.\d+', ' ', s)
    # 6. HDFS block IDs (blk_ followed by numbers/minus)
    s = re.sub(r'blk_-?\d+', ' ', s)
    # 7. Remaining numbers
    s = re.sub(r'\b\d+\b', ' ', s)
    # 8. Normalise whitespace, lower
    s = re.sub(r'\s+', ' ', s).strip().lower()

    return s if s else 'system event'


def encode_string_vocabulary(vocab_strings: list, batch_size=512):
    """
    Encodes a list of unique cleaned strings using all-MiniLM-L6-v2.
    Returns a float32 numpy matrix of shape (len(vocab_strings), 384).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Initializing SentenceTransformer on {device}...")
    encoder = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    
    print(f"[*] Encoding {len(vocab_strings):,} unique strings...")
    embeddings = encoder.encode(
        vocab_strings,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True
    )
    return embeddings.astype(np.float32)
