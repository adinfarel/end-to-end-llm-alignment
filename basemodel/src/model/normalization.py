'''
src/model/normalization.py

Normalization layer for the model with RMSNorm
'''

import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, embedding_dim, eps=1e-8):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.ones(embedding_dim))
        self.eps = eps
        
    def forward(self, x: torch.tensor):
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x_norm = x * rms
        return x_norm * self.weight