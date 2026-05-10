'''
src/model/gpt.py

This is cinema, this is the main idea of this project, all of logic is in here so time to shine >.<
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

from basemodel.src.model.embedding import Embedding
# from basemodel.src.model.pos_enc import LearnedPositionalEnc
from basemodel.src.model.block import Block
from basemodel.src.model.normalization import RMSNorm
from utils.common import load_yaml

@dataclass
class GPTConfig:
    vocab_size: int
    embedding_dim: int
    batch_size: int
    n_head: int
    learning_rate: float
    block_size: int
    dropout: float
    n_blocks: int
    
    @classmethod
    def config(cls, yaml_path: str):
        cfg = load_yaml(yaml_path)
        training = cfg['models']['training']
        eval = cfg['models']['eval']
        return cls(
            vocab_size=cfg['tokenizer']['vocab_size'],
            embedding_dim=training['embedding_dim'],
            batch_size=training['batch_size'],
            n_head=training['n_head'],
            learning_rate=training['learning_rate'],
            block_size=training['block_size'],
            dropout=training['dropout'],
            n_blocks=training['n_blocks'],
        )


class AlmondGPTModel(nn.Module):
    def __init__(self, config_path: str, vocab_size: int = None):
        super().__init__()
        self.config = GPTConfig.config(config_path)
        self.vocab_size = vocab_size if vocab_size is not None else self.config.vocab_size
        self.embedding = Embedding(self.vocab_size, self.config.embedding_dim)
        self.blocks = nn.ModuleList(
            [Block(n_embd=self.config.embedding_dim, n_heads=self.config.n_head,
                    dropout=self.config.dropout) for _ in range(self.config.n_blocks)]
        )
        self.ln_f = RMSNorm(self.config.embedding_dim)
        self.lm_head = nn.Linear(self.config.embedding_dim, self.vocab_size)
    
    def forward(self, x, targets=None, use_cache=False):
        B, T = x.shape
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x, use_cache=use_cache)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(-1, C)
            loss = F.cross_entropy(logits, targets.view(-1))
        
        return logits, loss

    def clear_kv_cache(self):
        for block in self.blocks:
            if hasattr(block.attn, "clear_cache"):
                block.attn.clear_cache()
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        self.clear_kv_cache()
        for i in range(max_new_tokens):
            curr_idx = idx if i == 0 else idx[:, -1:]
            
            logits, _ = self(curr_idx, use_cache=True)
            logits = logits[:, -1, :] # Last Token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
            
        return idx