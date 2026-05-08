'''
src/model/multiheadattn.py

Core idea of this model is Attention Mechanism
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from basemodel.src.model.pos_enc import RotaryPositionalEncoding

class SingleHeadAttention(nn.Module):
    def __init__(self, n_embd, head_size, dropout, block_size):
        super().__init__()
        self.head_size = head_size
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # Activate this if u want to implement dense attention
        # self.register_buffer(
        #     "tril",
        #     torch.tril(torch.ones(block_size, block_size))
        # )
        self.dropout = dropout
        self.rope = RotaryPositionalEncoding(head_size)
    
    def forward(self, x: torch.tensor):
        B, T, C = x.shape
        key = self.rope(self.key(x))
        query = self.rope(self.query(x))
        value = self.value(x)
        
        out = F.scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            is_causal=True,
            dropout_p=self.dropout.p if self.training else 0.0,
        )
        # attn = query @ key.transpose(-2, -1) * (self.head_size ** -0.5)
        
        # wei = attn.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        # wei = F.softmax(wei, dim=-1)
        # wei = self.dropout(wei)
        
        # out = wei @ value
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embd, dropout, block_size):
        super().__init__()
        self.heads = nn.ModuleList(
            [SingleHeadAttention(
                n_embd=n_embd,
                head_size=head_size,
                dropout=dropout,
                block_size=block_size
            ) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.tensor):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class GroupQueryAttention(nn.Module):
    def __init__(self, n_embd, n_q_head, n_kv_heads, dropout):
        super().__init__()
        assert n_embd % n_q_head == 0, "n_q_head must be divide n_embd with result = 0"
        assert n_q_head % n_kv_heads == 0, "n_kv_heads must be divide n_q_head with result = 0"
        
        self.n_embd = n_embd
        self.n_q_head = n_q_head
        self.n_kv_heads = n_kv_heads
        self.head_size = n_embd // n_q_head
        self.num_queries_per_kv_head = n_q_head // n_kv_heads
        self.dropout = dropout
        
        # Attention projections
        self.query = nn.Linear(n_embd, n_q_head * self.head_size, bias=False)
        self.key = nn.Linear(n_embd, n_kv_heads * self.head_size, bias=False)
        self.value = nn.Linear(n_embd, n_kv_heads * self.head_size, bias=False)
        self.proj = nn.Linear(n_q_head * self.head_size, n_embd, bias=False)
        
        # RoPE
        self.rope = RotaryPositionalEncoding(self.head_size)
    
    def forward(self, x: torch.tensor):
        B, T, C = x.shape
        
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        Q = Q.view(B, T, self.n_q_head, self.head_size).transpose(1, 2) # (B, n_q_head, T, head_size)
        
        K = K.view(B, T, self.n_kv_heads, self.head_size).transpose(1, 2) # (B, n_kv_heads, T, head_size)
        V = V.view(B, T, self.n_kv_heads, self.head_size).transpose(1, 2) # (B, n_kv_heads, T, head_size)
        
        Q = self.rope(Q)
        K = self.rope(K)
        
        K = K.repeat_interleave(self.num_queries_per_kv_head, dim=1)
        V = V.repeat_interleave(self.num_queries_per_kv_head, dim=1)
        
        out = F.scaled_dot_product_attention(
            query=Q,
            key=K,
            value=V,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        
        out = out.transpose(1, 2).contiguous()
        
        out = out.view(B, T, self.n_q_head * self.head_size)
        out = self.proj(out)
        
        if self.training and self.dropout > 0.0:
            out = F.dropout(out, p=self.dropout)
        
        return out