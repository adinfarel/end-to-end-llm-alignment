'''
src/model/utils.py

Utility for help all progress at main file gpt.py
'''

import torch
import torch.nn as nn
import numpy as np

def get_batch(data, batch_size, block_size):
    """
    Generate a batch of data for training.
    """
    ix = torch.randint(len(data) - block_size, (batch_size,))
    
    x = torch.stack([torch.tensor(data[i:i+block_size], dtype=torch.long) for i in ix])
    y = torch.stack([torch.tensor(data[i+1:i+block_size+1], dtype=torch.long) for i in ix])
    
    return x, y

@torch.no_grad()
def eval_loss(model: nn.Module, data: np.ndarray, config, device='cpu'):
    model.eval()
    losses = []
    for _ in range(config['eval']['eval_iters']):
        xb, yb = get_batch(data=data, batch_size=config['training']['batch_size'], block_size=config['training']['block_size'])
        xb, yb = xb.to(device=device), yb.to(device=device)
        logits, loss = model(xb, yb, use_cache=False)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)