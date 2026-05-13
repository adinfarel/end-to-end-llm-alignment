'''
src/data/datasets.py

Tokenized, add padding, and processed data for training >.<
'''

import torch
import os
from basemodel.src.tokenizer.bpe import AlmondTokenizerGPT
from torch.utils.data import Dataset

class InstructionDataset(Dataset):
    def __init__(self, data_list: list, tokenizer: AlmondTokenizerGPT) -> None:
        self.data = data_list # List of raw text data samples
        self.samples = []
        append = self.samples.append
        
        assistant_tag = "<|assistant|>"
        
        for text in data_list:
            assistant_pos = text.find(assistant_tag)
            
            if assistant_pos == -1:
                raise ValueError(f"Missing '{assistant_tag}' in sample:\n{text[:200]}")
            
            response_start = assistant_pos + len(assistant_tag)
            
            if text[response_start:response_start + 2] == '\r\n':
                response_start += 2
            elif text[response_start:response_start + 1] == '\n':
                response_start += 1
            
            prompt_text = text[:response_start]
            
            input_ids = tokenizer.encode(text)
            prompt_len = len(tokenizer.encode(prompt_text))
            
            append({
                "input_ids": input_ids,
                "prompt_len": prompt_len,
            })
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> list:
        return self.samples[idx]

def collate_fn(
    batch: list,
    pad_token_id: int,
    max_length: int,
    ignore_index: int = -100,
):
    batch_max_length = max(len(x['input_ids']) + 1 for x in batch)
    batch_max_length = min(batch_max_length, max_length + 1) # Ensure batch_max_length does not exceed max_length + 1
    
    input_lst, target = [], []
    append = input_lst.append; append_target = target.append # Optimization: Cache the append methods to avoid attribute lookups in the loop
    
    for item in batch:
        new_item = item['input_ids'].copy() # Create a copy of the item to avoid modifying the original list
        prompt_len = item['prompt_len']
        
        if len(new_item) > max_length:
            new_item = new_item[:max_length] # Truncate if exceeds max_length
        
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        inputs = torch.tensor(padded[:-1], dtype=torch.long) # Exclude the last token for inputs
        targets = torch.tensor(padded[1:], dtype=torch.long) # Exclude the first token for targets (shifted by one)
        
        targets[targets == pad_token_id] = ignore_index # Set padding tokens to ignore_index for loss calculation
        
        mask_until = max(prompt_len - 1, 0)
        targets[:mask_until] = ignore_index # Mask the prompt tokens in targets to ignore_index

        append(inputs)
        append_target(targets)
    
    input_tensor = torch.stack(input_lst) # Stack the input tensors into a single tensor and move to the specified device
    target_tensor = torch.stack(target) # Stack the target tensors into a single tensor and move to the specified device
    return input_tensor, target_tensor