# models/multi_view_detector.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiViewDetector(nn.Module):
    """
    v12 混合多视角架构 (Multi-View Architecture) - 语义视角特征提取器
    核心范式：放弃自回归与单中心超球体，转向监督对比学习 (InfoNCE)。
    通过 Transformer 提取时序语义特征，并投影到 L2 归一化的隐空间，
    为后续的 KNN 局部密度计算 (语义视角) 提供高质量的表征。
    """
    def __init__(self, vocab_size, embed_dim=64, n_heads=4, num_layers=2, proj_dim=32, max_len=10, pad_idx=0):
        super().__init__()
        self.pad_idx = pad_idx
        
        # 1. 嵌入层与位置编码
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.pos_encoder = nn.Parameter(torch.randn(1, max_len, embed_dim) * 0.02)
        
        # 2. Transformer 编码器 (捕获日志时序因果依赖)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim*4,
            dropout=0.1, batch_first=True, activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. 投影头 (Projection Head) - 对比学习必备组件
        # 将高维特征映射到低维流形，并使用 LayerNorm 稳定训练
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim)
        )

    def forward(self, x):
        # 生成 Padding Mask (True 表示需要被 mask 掉的位置)
        pad_mask = (x == self.pad_idx) 
        
        # 嵌入 + 位置编码
        x = self.embedding(x) + self.pos_encoder[:, :x.size(1), :]
        
        # Transformer 编码
        features = self.transformer_encoder(x, src_key_padding_mask=pad_mask)
        
        # 稳定的 Masked Mean Pooling (避免 PAD token 污染全局表征)
        features = features.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        valid_counts = (~pad_mask).sum(dim=1, keepdim=True).float().clamp(min=1.0)
        pooled = features.sum(dim=1) / valid_counts
        
        # 投影并 L2 归一化 (InfoNCE 损失函数要求特征在单位超球面上)
        z = self.projection(pooled)
        z = F.normalize(z, p=2, dim=1) 
        return z