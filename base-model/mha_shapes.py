"""
Multi-Head Attention with explicit matrix math and shapes.
"""

import os 
import math
import torch
from multi_head import MultiHeadSelfAttention

OUT_TXT = os.path.join(os.path.dirname(__name__), 'out', 'mha_shapes.txt')

def log(s):
    print(s)
    with open(OUT_TXT, 'a') as f:
        f.write(s + "\n")

if __name__ == "__main__":
    # Reset file
    os.makedirs(os.path.dirname(OUT_TXT), exist_ok=True)
    open(OUT_TXT, 'r').close()
    
    B, T, d_model, n_head = 1, 5, 12, 3
    d_head = d_model // n_head
    x = torch.randn(B, T, d_model)
    attn = MultiHeadSelfAttention(d_model, n_head, trace_shapes=True)
    
    log(f"Input x:              {tuple(x.shape)} = (B, T, d_model)")
    qkv = attn.qkv(x)
    log(f"Linear qkv(x):        {tuple(qkv.shape)} = (B, T, 3*d_model)")
    
    qkv = qkv.view(B, T, 3, n_head, d_head)
    log(f"View to 5D:           {tuple(qkv.shape)} = (B, T, 3, heads, d_head)")
    
    q, k, v = qkv.unbind(dim=2)
    log(f"q,k,v split:          q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)}")
    
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    log(f"Transpose heads:      q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)} = (B, heads, T, d_head)")
    
    scale = 1.0 / math.sqrt(d_head)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    log(f"scores q@k.T:         {tuple(scores.shape)} = (B, n_heads, T, T)")
    
    weights = torch.softmax(scores, dim=-1)
    log(f"softmax (weights):    {tuple(scores.shape)} = (B, n_heads, T, T)")
    
    ctx = torch.matmul(weights, v)
    log(f"context @v:           {tuple(ctx.shape)} = (B, heads, T, n_heads)")
    
    out = ctx.transpose(1, 2).contiguous().view(B, T, d_model)
    log(f"merge heads:          {tuple(out.shape)} = (B, T, d_model)")
    
    out = attn.proj(out)
    log(f"final proj:           {tuple(out.shape)} = (B, T, d_model)")
    
    log(f"\nLegend")
    log("B = Batch, T = Token Sequence, d_model = Embedding size, heads = n_heads, d_head = d_model / n_heads")
    log("qkv(x) is a single Linear producing [Q|K|V]: we reshape then split into q,k,v")