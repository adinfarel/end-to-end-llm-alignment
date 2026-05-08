'''
src/model/ffwd.py

To get non-linear output at 4 times space from n_embed and squash (projection) again to n_embd while to keep important non-linear things (relevant).
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.ffwd = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4),
            nn.GELU(),
            nn.Linear(n_embd * 4, n_embd),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.ffwd(x)

class SwiGLUFFN(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.n_embd = int(n_embd * (8/3))
        self.gate = nn.Linear(n_embd, self.n_embd, bias=False)
        self.value = nn.Linear(n_embd, self.n_embd, bias=False)
        self.down_proj = nn.Linear(self.n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        value = self.value(x)
        gate = F.silu(self.gate(x)) # Formula: x * sigmoid(x) = x * (1 / (1 + exp(-x))) = x / (1 + exp(-x))
        hidden = value * gate
        out = self.down_proj(hidden)
        return self.dropout(out)