'''
src/dpo/dpo.py

The main idea of this alignment Direct Preference Optimization,
Why i'm choose DPO? easy implement but hard mathematics (I think so... :V), not need many models as a (RM or Critic) we just need Reference Model >.<
'''

import torch
import torch.nn.functional as F
import torch.nn as nn

def get_batch_logps(
    logits: torch.tensor,
    labels: torch.tensor,
    ignore_index: int = -100
) -> torch.tensor:
    '''
    Calculate total log-probability by a full-sentence
    '''
    logits = logits[:, :-1, :]
    labels = labels[:, 1:].clone()
    
    loss_mask = (labels != ignore_index) # Masking to ignore padding and prompt 
    
    labels[labels == ignore_index] = 0
    
    per_token_logits = F.log_softmax(logits, dim=-1)
    per_token_logs = torch.gather(per_token_logits, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    
    per_token_logs *= loss_mask
    
    return per_token_logs.sum(-1)

class DPOLoss(nn.Module):
    def __init__(self, beta: float = 0.1):
        super().__init__()
        self.beta = beta
    
    def forward(
        self,
        policy_chosen_logits: torch.Tensor,
        policy_rejected_logits: torch.Tensor,
        reference_chosen_logits: torch.Tensor,
        reference_rejected_logits: torch.Tensor,
        chosen_labels: torch.Tensor,
        rejected_labels: torch.Tensor,
    ):
        pi_chosen_logits = get_batch_logps(policy_chosen_logits, chosen_labels)
        pi_rejected_logits = get_batch_logps(policy_rejected_logits, rejected_labels)
        
        with torch.no_grad():
            ref_chosen_logits = get_batch_logps(reference_chosen_logits.detach(), chosen_labels)
            ref_rejected_logits = get_batch_logps(reference_rejected_logits.detach(), rejected_labels)
            
        pi_logratios = pi_chosen_logits - pi_rejected_logits
        ref_logratios = ref_chosen_logits - ref_rejected_logits
        
        logits = pi_logratios - ref_logratios
        loss = -F.logsigmoid(self.beta * logits)
        
        return loss.mean()
        