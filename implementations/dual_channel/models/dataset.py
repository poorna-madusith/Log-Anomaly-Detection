"""
PyTorch Dataset and DataLoader (FYP Final Implementation)
========================================================================
Memory-efficient data ingestion.
Upgrades:
1. Memory-Mapped Loading (mmap_mode='r') to prevent duplicating 
   multi-gigabyte embedding matrices across train/val/test splits.
2. Supports Index 0 as true <PAD> vector (all zeros).
3. Dynamic positive class weight calculation for class imbalance.
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class PerLineLogDataset(Dataset):
    def __init__(self, npz_path: str, string_matrix_path: str):
        with np.load(npz_path) as data:
            self.X_struct = data["X_struct"]
            self.X_sem_idx = data["X_sem_idx"]
            self.y = data["y"]
        
        # Memory-mapped loading to prevent duplicating multi-gigabyte matrices across splits
        self.string_matrix = np.load(string_matrix_path, mmap_mode='r')
        
    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, idx):
        struct_seq = torch.from_numpy(self.X_struct[idx]).long()
        sem_indices = self.X_sem_idx[idx]
        
        # Fast memory-mapped slice onto float32 tensor
        sem_seq = torch.from_numpy(self.string_matrix[sem_indices].copy()).float()
        label = torch.tensor(self.y[idx], dtype=torch.float32)
        
        return struct_seq, sem_seq, label

    def get_pos_weight(self):
        """Calculates exact mathematical inverse frequency for BCEWithLogitsLoss."""
        n_pos = np.sum(self.y == 1)
        n_neg = np.sum(self.y == 0)
        if n_pos == 0 or n_neg == 0:
            return torch.tensor([1.0], dtype=torch.float32)
        return torch.tensor([n_neg / n_pos], dtype=torch.float32)

def create_dataloader(npz_path: str, string_matrix_path: str, batch_size: int = 256, shuffle: bool = True):
    dataset = PerLineLogDataset(npz_path, string_matrix_path)
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=0, 
        pin_memory=torch.cuda.is_available()
    )
    return loader, dataset
