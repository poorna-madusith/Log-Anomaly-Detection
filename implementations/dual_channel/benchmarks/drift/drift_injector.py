"""
Semantic Drift Injector (FYP Final Implementation)
========================================================================
Simulates CI/CD developer log drift under the strict token convention:
- ID 0: <PAD> (Strictly preserved, NEVER mutated)
- ID 1: <OOV> (Target token for drifted/mutated templates)
- ID >= 2: Known Drain templates (Mutated to ID 1 with probability drift_ratio)

The semantic embeddings (SentenceTransformer vectors) remain untouched,
faithfully simulating the reality that developer rephrasings alter template syntax
while preserving the underlying semantic meaning of the system event.
"""

import torch

def inject_structural_drift(struct_seqs: torch.Tensor, drift_ratio: float = 0.0, oov_token_id: int = 1) -> torch.Tensor:
    """
    Vectorized drift injection on a tensor of structural event IDs.
    
    Args:
        struct_seqs (torch.Tensor): Shape [Batch, Seq_Len] containing integer structural IDs.
        drift_ratio (float): Percentage of real tokens to mutate to unseen (e.g., 0.10 for 10% drift).
        oov_token_id (int): ID representing an unseen/drifted template (strictly 1).
        
    Returns:
        torch.Tensor: Drifted structural tensor where mutated tokens are assigned ID 1.
    """
    if drift_ratio <= 0.0:
        return struct_seqs.clone()
        
    drifted = struct_seqs.clone()
    # Only mutate real structural tokens (ID >= 2), NEVER mutate <PAD> (ID 0) or existing <OOV> (ID 1)
    real_token_mask = (drifted > 1)
    
    # Bernoulli sampling across candidate positions
    random_prob = torch.rand_like(drifted, dtype=torch.float32)
    drift_mask = (random_prob < drift_ratio) & real_token_mask
    
    # Mutate selected positions to strictly <OOV> token ID (1)
    drifted[drift_mask] = oov_token_id
    return drifted
