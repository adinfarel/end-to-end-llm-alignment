'''
src/model/ffwd.py

To get non-linear output at 4 times space from n_embed and squash (projection) again to n_embd while to keep important non-linear things (relevant).
'''

import torch
import torch.nn as nn

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