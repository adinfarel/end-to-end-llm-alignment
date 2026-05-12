'''
src/data/datasets.py

Tokenized, add padding, and processed data for training >.<
'''

import torch
import os
from basemodel.src.tokenizer.bpe import AlmondTokenizerGPT
from torch.utils.data import Dataset

class InstructionDataset(Dataset):
    def __init__(self, data_list: list, tokenizer: AlmondTokenizerGPT, device: torch.device) -> None:
        self.data = data_list # List of raw text data samples
        self.encoded_texts = []
        
        for text in self.data:
            tokenized_data = tokenizer.encode(text)
            self.encoded_texts.append(tokenized_data) 
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> list:
        return self.encoded_texts[idx]

def collate_fn(
    batch: list,
    pad_token_id: int,
    max_length: int,
    ignore_index: int = -100,
    device: torch.device = 'cpu'
):
    batch_max_length = max(len(x) + 1 for x in batch)
    batch_max_length = min(batch_max_length, max_length + 1) # Ensure batch_max_length does not exceed max_length + 1
    
    input_lst, target = [], []
    append = input_lst.append; append_target = target.append # Optimization: Cache the append methods to avoid attribute lookups in the loop
    
    for item in batch:
        new_item = item.copy() # Create a copy of the item to avoid modifying the original list
        
        if len(new_item) > max_length:
            new_item = new_item[:max_length] # Truncate if exceeds max_length
        
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        inputs = torch.tensor(padded[:-1], dtype=torch.long, device=device) # Exclude the last token for inputs
        targets = torch.tensor(padded[1:], dtype=torch.long, device=device) # Exclude the first token for targets (shifted by one)
        
        targets[targets == pad_token_id] = ignore_index # Set padding tokens to ignore_index for loss calculation

        append(inputs)
        append_target(targets)
    
    input_tensor = torch.stack(input_lst).to(device) # Stack the input tensors into a single tensor and move to the specified device
    target_tensor = torch.stack(target).to(device) # Stack the target tensors into a single tensor and move to the specified device
    return input_tensor, target_tensor