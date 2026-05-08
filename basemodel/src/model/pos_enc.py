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
        inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim)) # (embedding_dim / 2)
        self.register_buffer("inv_freq", inv_freq)
    
    def forward(self, x: torch.tensor):
        T, C = x.shape[-2], x.shape[-1]
        assert C == self.dim
        l = torch.arange(T, dtype=self.inv_freq.dtype) # [0, 1, 2, ..., L]
        theta = torch.einsum("i,j->ij", l, self.inv_freq) # (l, inv_freq)
        hat_theta = torch.cat([theta, theta], axis=-1) # (l, inv_freq + inv_freq)
        sin = torch.sin(hat_theta) # (l, hat_theta)
        cos = torch.cos(hat_theta) # (l, hat_theta)
        xu, xd = x[..., : C // 2], x[..., C // 2 :]
        hatx = torch.cat([-xd, xu], axis=-1)
        return x * cos + hatx * sin