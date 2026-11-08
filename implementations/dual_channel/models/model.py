"""
Dual-Channel Fusion Bi-LSTM Architecture (FYP Final Implementation)
========================================================================
Core model architecture with:
1. Padding Masking: Position 0 is reserved for <PAD>. Padded timesteps are 
   masked in attention so they contribute 0 weight to the context vector.
2. Structural Dropout: Randomly masks structural representation during training 
   to enforce fallback to semantic representations under OOV drift.
3. Softmax Attention: Outputs normalized attention weights for SRE heatmaps.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DualChannelBiLSTM(nn.Module):
    def __init__(
        self,
        vocab_size=10000,
        struct_embed_dim=64,
        struct_hidden_dim=64,
        sem_input_dim=384,
        sem_proj_dim=128,
        sem_hidden_dim=128,
        struct_dropout_prob=0.3,
        fc_hidden_dim=64,
        dropout_prob=0.3
    ):
        super(DualChannelBiLSTM, self).__init__()
        self.struct_dropout_prob = struct_dropout_prob
        
        # Channel 1: Structural (padding_idx=0 ensures embedding for <PAD> is always 0)
        self.struct_embedding = nn.Embedding(vocab_size, struct_embed_dim, padding_idx=0)
        self.struct_lstm = nn.LSTM(struct_embed_dim, struct_hidden_dim, num_layers=1, bidirectional=True, batch_first=True)
        self.struct_dim = struct_hidden_dim * 2
        
        # Channel 2: Semantic
        self.sem_projection = nn.Sequential(
            nn.Linear(sem_input_dim, sem_proj_dim),
            nn.LayerNorm(sem_proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )
        self.sem_lstm = nn.LSTM(sem_proj_dim, sem_hidden_dim, num_layers=1, bidirectional=True, batch_first=True)
        self.sem_dim = sem_hidden_dim * 2
        
        # Fusion & Attention
        self.fused_dim = self.struct_dim + self.sem_dim
        self.attention_net = nn.Sequential(
            nn.Linear(self.fused_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1, bias=False)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(self.fused_dim, fc_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(fc_hidden_dim, 1)
        )

    def forward(self, struct_seq, sem_seq):
        batch_size, seq_len = struct_seq.size()
        
        # True sequence lengths based on non-padding tokens (PAD = 0)
        lengths = (struct_seq != 0).sum(dim=1).clamp(min=1).cpu()
        
        embed_struct = self.struct_embedding(struct_seq)
        packed_struct = nn.utils.rnn.pack_padded_sequence(
            embed_struct, lengths, batch_first=True, enforce_sorted=False
        )
        packed_out_struct, _ = self.struct_lstm(packed_struct)
        out_struct, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out_struct, batch_first=True, total_length=seq_len
        )
        
        # Structural Dropout (forces reliance on semantics)
        if self.training and self.struct_dropout_prob > 0.0:
            mask = (torch.rand(batch_size, 1, 1, device=struct_seq.device) >= self.struct_dropout_prob).float()
            out_struct = (out_struct * mask) / (1.0 - self.struct_dropout_prob)
            
        proj_sem = self.sem_projection(sem_seq)
        packed_sem = nn.utils.rnn.pack_padded_sequence(
            proj_sem, lengths, batch_first=True, enforce_sorted=False
        )
        packed_out_sem, _ = self.sem_lstm(packed_sem)
        out_sem, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out_sem, batch_first=True, total_length=seq_len
        )
        
        fused = torch.cat([out_struct, out_sem], dim=-1)
        
        # Compute Attention Energy Scores
        attn_scores = self.attention_net(fused).squeeze(-1) # [Batch, SeqLen]
        
        # Apply Padding Mask: ONLY positions where struct_seq == 0 (<PAD>) are masked.
        # ID 1 (<OOV>) is a real event and is NEVER masked!
        pad_mask = (struct_seq == 0)
        if pad_mask.any():
            attn_scores = attn_scores.masked_fill(pad_mask, -1e4)
            
        attention_weights = F.softmax(attn_scores, dim=-1)
        if pad_mask.any():
            attention_weights = attention_weights.masked_fill(pad_mask, 0.0)
            weight_sum = attention_weights.sum(dim=-1, keepdim=True)
            attention_weights = attention_weights / (weight_sum + 1e-9)

        context = torch.bmm(attention_weights.unsqueeze(1), fused).squeeze(1)
        
        logits = self.classifier(context).squeeze(-1)
        return logits, attention_weights
