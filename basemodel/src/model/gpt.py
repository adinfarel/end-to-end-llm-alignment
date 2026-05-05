'''
src/model/gpt.py

This is cinema, this is the main idea of this project, all of logic is in here so time to shine >.<
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

from basemodel.src.model.embedding import Embedding
from basemodel.src.model.pos_enc import LearnedPositionalEnc
from basemodel.src.model.block import Block
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
            model_save_path=training['model_save_path'],
            training=training['training'],
            max_iters=eval['max_iters'],
            eval_interval=eval['eval_interval'],
            eval_iters=eval['eval_iters']
        )


class AlmondGPTModel(nn.Module):
    def __init__(self, config_path: str):
        super().__init__()
        self.config = GPTConfig.config(config_path)
        self.vocab_size = self.config.vocab_size
        self.embedding = Embedding(self.vocab_size, self.config.embedding_dim)
        self.pos_enc = LearnedPositionalEnc(self.config.block_size, self.config.block_size)
        self.blocks = nn.Sequential(
            *[Block(n_embd=self.config.embedding_dim, n_heads=self.config.n_head,
                    dropout=self.config.dropout, block_size=self.config.block_size) for _ in range(self.config.n_blocks)]
        )
        self.ln_f = nn.LayerNorm(self.config.embedding_dim)
        self.lm_head = nn.Linear(self.config.embedding_dim, self.vocab_size)
    
    def forward(self, x, targets=None):
        B, T = x.shape
        x = self.embedding(x)
        x = self.pos_enc(x)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(-1, C)
            loss = F.cross_entropy(logits, targets.view(-1))
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            curr_idx = idx[:, -self.config.block_size:]
            
            logits, _ = self(curr_idx)
            logits = logits[:, -1, :] # Last Token
            probs = F.softmax(logits, dim=-1, keepdim=True)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((curr_idx, next_token), dim=1)
            
        return idx