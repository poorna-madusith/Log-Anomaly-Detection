"""
Baseline Architectures for Controlled Channel Ablation Study
========================================================================
Implements single-channel baselines matching DualChannelBiLSTM layer-for-layer:
1. StructuralBaselineBiLSTM (DeepLog / Drain-LSTM equivalent):
   - Channel 1 in isolation: Embedding(vocab_size, 64, padding_idx=0) -> BiLSTM(64, 64) -> Attention -> MLP
2. SemanticBaselineBiLSTM (Semantic-Only Ablation):
   - Channel 2 in isolation: Linear(384, 128) -> BiLSTM(128, 128) -> Attention -> MLP
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class StructuralBaselineBiLSTM(nn.Module):
    """DeepLog / Drain-LSTM Equivalent: Structural Channel in isolation."""
    def __init__(
        self,
        vocab_size=10000,
        embed_dim=64,
        hidden_dim=64,
        fc_hidden_dim=64,
        dropout_prob=0.3
    ):
        super(StructuralBaselineBiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=1, bidirectional=True, batch_first=True)
        self.out_dim = hidden_dim * 2 # 128
        
        self.attention_net = nn.Sequential(
            nn.Linear(self.out_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1, bias=False)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(self.out_dim, fc_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(fc_hidden_dim, 1)
        )

    def forward(self, struct_seq):
        batch_size, seq_len = struct_seq.size()
        lengths = (struct_seq != 0).sum(dim=1).clamp(min=1).cpu()
        
        embed = self.embedding(struct_seq)
        packed = nn.utils.rnn.pack_padded_sequence(embed, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=seq_len)
        
        attn_scores = self.attention_net(out).squeeze(-1)
        pad_mask = (struct_seq == 0)
        if pad_mask.any():
            attn_scores = attn_scores.masked_fill(pad_mask, -1e4)
            
        weights = F.softmax(attn_scores, dim=-1)
        if pad_mask.any():
            weights = weights.masked_fill(pad_mask, 0.0)
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-9)
            
        context = torch.bmm(weights.unsqueeze(1), out).squeeze(1)
        logits = self.classifier(context).squeeze(-1)
        return logits, weights


class SemanticBaselineBiLSTM(nn.Module):
    """Semantic-Only Ablation: Semantic Channel in isolation."""
    def __init__(
        self,
        sem_input_dim=384,
        sem_proj_dim=128,
        hidden_dim=128,
        fc_hidden_dim=64,
        dropout_prob=0.3
    ):
        super(SemanticBaselineBiLSTM, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(sem_input_dim, sem_proj_dim),
            nn.LayerNorm(sem_proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )
        self.lstm = nn.LSTM(sem_proj_dim, hidden_dim, num_layers=1, bidirectional=True, batch_first=True)
        self.out_dim = hidden_dim * 2 # 256
        
        self.attention_net = nn.Sequential(
            nn.Linear(self.out_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1, bias=False)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(self.out_dim, fc_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(fc_hidden_dim, 1)
        )

    def forward(self, struct_seq, sem_seq):
        batch_size, seq_len = struct_seq.size()
        lengths = (struct_seq != 0).sum(dim=1).clamp(min=1).cpu()
        
        proj = self.projection(sem_seq)
        packed = nn.utils.rnn.pack_padded_sequence(proj, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=seq_len)
        
        attn_scores = self.attention_net(out).squeeze(-1)
        pad_mask = (struct_seq == 0)
        if pad_mask.any():
            attn_scores = attn_scores.masked_fill(pad_mask, -1e4)
            
        weights = F.softmax(attn_scores, dim=-1)
        if pad_mask.any():
            weights = weights.masked_fill(pad_mask, 0.0)
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-9)
            
        context = torch.bmm(weights.unsqueeze(1), out).squeeze(1)
        logits = self.classifier(context).squeeze(-1)
        return logits, weights
