'''
src/model/block.py

A single transforme block that contains all layer of transformers
'''

import torch
import torch.nn as nn
from basemodel.src.model import (
    multiheadattn, ffwd
)

class Block(nn.Module):
    def __init__(self, n_embd, n_heads, dropout, block_size):
        super().__init__()
        assert n_embd % n_heads == 0, "n_heads must be divide n_embd with result = 0"
        head_size = n_embd // n_heads
        self.attn = multiheadattn.MultiHeadAttention(num_heads=n_heads, head_size=head_size, n_embd=n_embd, dropout=dropout, block_size=block_size)
        self.ffwd = ffwd.FeedForward(n_embd=n_embd, dropout=dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x