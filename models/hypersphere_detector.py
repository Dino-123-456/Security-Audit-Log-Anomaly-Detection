# models/hypersphere_detector.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class LogHypersphereDetector(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, n_heads=4, num_layers=2, proj_dim=32, max_len=10, pad_idx=0):
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.pos_encoder = nn.Parameter(torch.randn(1, max_len, embed_dim) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim*4,
            dropout=0.1, batch_first=True, activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 投影头 (Projection Head)
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim)
        )

    def forward(self, x):
        pad_mask = (x == self.pad_idx) 
        x = self.embedding(x) + self.pos_encoder[:, :x.size(1), :]
        features = self.transformer_encoder(x, src_key_padding_mask=pad_mask)
        
        # 稳定的 Masked Mean Pooling
        features = features.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        valid_counts = (~pad_mask).sum(dim=1, keepdim=True).float().clamp(min=1.0)
        pooled = features.sum(dim=1) / valid_counts
        
        z = self.projection(pooled)
        z = F.normalize(z, p=2, dim=1) # L2 归一化
        return z