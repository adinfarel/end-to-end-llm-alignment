import torch

def causal_mask(T: int, device=None):
    """
    Returned a bool mask, where masked it is True means
    """
    m = torch.triu(torch.ones((T, T)), dtype=torch.bool, diagonal=1)
    return m.view(1,1,T,T)