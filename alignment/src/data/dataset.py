'''
src/data/dataset.py

Load datasets.json for Direct Preference Optimization Training 
'''

import os
import torch
from dataclasses import dataclass
from utils.common import load_yaml, load_json
from basemodel.src.tokenizer.bpe import AlmondTokenizerGPT

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

# -------------------------
DPO_DATASET_PATH = 'alignment/data/raw/dpo_dataset.json'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# -------------------------

class DpoDatasets(Dataset):
    def __init__(self, data: dict, tokenizer: AlmondTokenizerGPT):
        self.data = data
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index: int):
        item = self.data[index]
        
        prompt = item['prompt']
        chosen = item['chosen']
        rejected = item['rejected']
        
        prompt_ids = self.tokenizer.encode(prompt + "\n")
        prompt_len = len(prompt_ids)
        
        chosen_text = prompt + "\n" + chosen + "<|endoftext|>"
        rejected_text = prompt + "\n" + rejected + "<|endoftext|>"
        
        chosen_ids = self.tokenizer.encode(chosen_text)
        reject_ids = self.tokenizer.encode(rejected_text)
        
        chosen_labels = [-100] * prompt_len + chosen_ids[prompt_len:]
        reject_labels = [-100] * prompt_len + reject_ids[prompt_len:]
        
        return {
            "chosen_input_ids": chosen_ids,
            "rejected_input_ids": reject_ids,
            "chosen_labels": chosen_labels,
            "rejected_labels": reject_labels
        }

def collate_fn(
    batch: list,
    pad_token_id: int,
    max_length: int
):
    chosen_batch = []
    reject_batch = []
    chosen_lbls_batch = []
    reject_lbls_batch = []
    append_chosen = chosen_batch.append; append_reject = reject_batch.append
    append_chsn_lbls = chosen_lbls_batch.append; append_rjct_lbls = reject_lbls_batch.append
    
    for item in batch:
        chosen_ids = item['chosen_input_ids'][:max_length]
        reject_ids = item['rejected_input_ids'][:max_length]
        chosen_lbls = item['chosen_labels'][:max_length]
        reject_lbls = item['rejected_labels'][:max_length]
        
        append_chosen(torch.tensor(chosen_ids, dtype=torch.long))
        append_reject(torch.tensor(reject_ids, dtype=torch.long))
        append_chsn_lbls(torch.tensor(chosen_lbls, dtype=torch.long))
        append_rjct_lbls(torch.tensor(reject_lbls, dtype=torch.long))
    
    chosen_batch = pad_sequence(
        sequences=chosen_batch,
        batch_first=True,
        padding_value=pad_token_id,
    )
    
    reject_batch = pad_sequence(
        sequences=reject_batch,
        batch_first=True,
        padding_value=pad_token_id
    )
    
    chosen_batch_labels = pad_sequence(
        sequences=chosen_lbls_batch,
        batch_first=True,
        padding_value=-100
    )
    
    reject_batch_labels = pad_sequence(
        sequences=reject_lbls_batch,
        batch_first=True,
        padding_value=-100
    )
    
    return {
        "chosen_input_ids": chosen_batch,
        "rejected_input_ids": reject_batch,
        "chosen_labels": chosen_batch_labels,
        "rejected_labels": reject_batch_labels
    }