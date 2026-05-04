'''
src/model/embedding.py

Add embedding for each token as a representation
'''

import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
    
    def forward(self, x: torch.tensor):
        return self.embedding(x)