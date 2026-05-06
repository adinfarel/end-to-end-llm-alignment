'''
src/model/pos_emb.py

Positional embedding each token to get interval between token >.<
'''

import torch
import torch.nn as nn

class LearnedPositionalEnc(nn.Module):
    def __init__(self, block_size: int, embedding_dim: int):
        super().__init__()
        self.block_size = block_size
        self.pos_emb = nn.Embedding(block_size, embedding_dim)
    
    def forward(self, x: torch.tensor):
        B, T, C = x.shape
        arange = torch.arange(T, device=x.device)
        return self.pos_emb(arange) + x

class RotaryPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim: int, base=10000.0):
        super().__init__()
        self.dim = embedding_dim
        inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim))
        self.register_buffer("inv_freq", inv_freq)
    
    def forward(self, x: torch.tensor):
        B, T, C = x.shape
        assert C == self.dim
        l = torch.arange(T, dtype=self.theta.dtype) # [0, 1, 2, ..., L]
        theta = torch.einsum("i,j->ij", l, self.theta)
        hat_theta = torch.cat([theta, theta], axis=-1)
        sin = torch.sin(hat_theta)
        cos = torch.cos(hat_theta)
        xu, xd = x[..., : C // 2], x[..., C // 2 :]
        hatx = torch.cat([-xd, xu], axis=-1)
        return x * cos + hatx * sin