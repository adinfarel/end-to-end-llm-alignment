'''
src/data/loader.py

Load data for training, validation, and testing.
and then plugin into DataLoader >.<.
'''

import os

import torch
torch.manual_seed(42) # For reproducibility
from torch.utils.data import DataLoader
from dataclasses import dataclass
from functools import partial

from torch.utils.data import DataLoader
from sftmodel.src.data.datasets import InstructionDataset, collate_fn
from basemodel.src.tokenizer.bpe import AlmondTokenizerGPT
from sftmodel.src.data.converter import load_txt_to_list, split_data
from utils.common import load_yaml

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@dataclass
class DataLoaderConfig:
    batch_size: int
    max_length: int
    raw_data_path: str
    train_ratio: float
    test_ratio: float
    shuffle: bool
    num_workers: int
    
    @classmethod
    def config(cls, config_path: str):
        cfg = load_yaml(config_path)['loader']
        return cls(
            batch_size=cfg['batch_size'],
            max_length=cfg['max_length'],
            raw_data_path=cfg['raw_save_path'],
            train_ratio=cfg['train_ratio'],
            test_ratio=cfg['test_ratio'],
            shuffle=cfg['shuffle'],
            num_workers=cfg['num_workers']
        )

def create_dataloaders(config: DataLoaderConfig, tokenizer: AlmondTokenizerGPT, collate_fn, device: torch.device):
    '''Create DataLoaders for training, validation, and testing >.<
    
    Args:
        config (DataLoaderConfig): Configuration for data loading.
        tokenizer (AlmondTokenizerGPT): Tokenizer to encode the text data.
        collate_fn: Function to collate batches of data.
        device (torch.device): Device to load the tensors onto.
    
    Returns:
        tuple: (train_loader, test_loader, val_loader)
    '''
    print(f"Loading raw data from {config.raw_data_path}...")
    data_list = load_txt_to_list(os.path.join(config.raw_data_path, 'instructions.txt'))
    
    print(f"Splitting data into train, test, and validation sets with ratios {config.train_ratio}, {config.test_ratio}, and {1 - config.train_ratio - config.test_ratio}...")
    train_data, test_data, val_data = split_data(data_list, config.train_ratio, config.test_ratio)
    
    print(f"Creating InstructionDataset for train, test, and validation sets...")
    train_dataset = InstructionDataset(train_data, tokenizer)
    test_dataset = InstructionDataset(test_data, tokenizer)
    val_dataset = InstructionDataset(val_data, tokenizer)
    
    PAD_TOKEN_ID = tokenizer.single_byte_size + tokenizer.SPECIAL_TOKEN.index('<|pad|>')
    
    print("Build collate_fn with partial to include pad_token_id, max_length, ignore_index, and device...")
    custome_collate = partial(
        collate_fn,
        pad_token_id=PAD_TOKEN_ID,
        max_length=config.max_length,
        ignore_index=-100,
    )
    
    print(f"Creating DataLoader for train, test, and validation sets...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        collate_fn=custome_collate,
        shuffle=config.shuffle,
        num_workers=config.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        collate_fn=custome_collate,
        shuffle=False, # we don shuffle test data
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        collate_fn=custome_collate,
        shuffle=False, # we don shuffle validation data
        num_workers=config.num_workers,
    )
    
    return train_loader, test_loader, val_loader

if __name__ == "__main__":
    config = DataLoaderConfig.config('sftmodel/config.yaml')
    tokenizer = AlmondTokenizerGPT('basemodel/config.yaml')
    train_loader, test_loader, val_loader = create_dataloaders(config, tokenizer, collate_fn, DEVICE)
    print(f"DataLoaders created successfully. Train batches: {len(train_loader)}, Test batches: {len(test_loader)}, Validation batches: {len(val_loader)} >.<")
    
    print("Iterating through one batch of the training DataLoader to verify...")
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        print(f"Batch {batch_idx + 1}:")
        print(f"Inputs shape: {inputs.shape}, Targets shape: {targets.shape}")
        if batch_idx == 0:
            print(f"Sample input IDs: {inputs[0].tolist()}")
            print(f"Sample target IDs: {targets[0].tolist()}")
        if batch_idx == 10: # Just check the first 10 batches
            break