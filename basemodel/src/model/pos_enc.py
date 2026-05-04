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
        arange = torch.arange(self.block_size, device=x.device)
        return self.pos_emb(arange) + x