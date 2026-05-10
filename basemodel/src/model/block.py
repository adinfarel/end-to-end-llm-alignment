'''
src/model/block.py

A single transforme block that contains all layer of transformers
'''

import torch
import torch.nn as nn
from basemodel.src.model import (
    multiheadattn, ffwd, normalization
)

class Block(nn.Module):
    def __init__(self, n_embd, n_heads, dropout):
        super().__init__()
        assert n_embd % n_heads == 0, "n_heads must be divide n_embd with result = 0"
        # head_size = n_embd // n_heads
        self.attn = multiheadattn.GroupQueryAttention(n_embd=n_embd, n_q_head=n_heads, n_kv_heads=n_heads//2, dropout=dropout)
        self.ffwd = ffwd.SwiGLUFFN(n_embd=n_embd, dropout=dropout)
        self.ln1 = normalization.RMSNorm(n_embd)
        self.ln2 = normalization.RMSNorm(n_embd)
    
    def forward(self, x, use_cache=False):
        x = x + self.attn(self.ln1(x), use_cache=use_cache)
        x = x + self.ffwd(self.ln2(x))
        return x