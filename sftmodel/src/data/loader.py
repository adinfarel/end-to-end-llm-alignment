'''
src/data/loader.py

Load data for training, validation, and testing.
and then plugin into DataLoader >.<.
'''

import torch
torch.manual_seed(42) # For reproducibility
from torch.utils.data import DataLoader
from dataclasses import dataclass
from functools import partial

from torch.utils.data import DataLoader
from sftmodel.src.data.datasets import InstructionDataset, collate_fn
from basemodel.src.tokenizer.bpe import AlmondTokenizerGPT
from data.converter import load_txt_to_list, split_data
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
            raw_data_path=cfg['raw_data_path'],
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
    data_list = load_txt_to_list(config.raw_data_path)
    
    print(f"Splitting data into train, test, and validation sets with ratios {config.train_ratio}, {config.test_ratio}, and {1 - config.train_ratio - config.test_ratio}...")
    train_data, test_data, val_data = split_data(data_list, config.train_ratio, config.test_ratio)
    
    print(f"Creating InstructionDataset for train, test, and validation sets...")
    train_dataset = InstructionDataset(train_data, tokenizer, device)
    test_dataset = InstructionDataset(test_data, tokenizer, device)
    val_dataset = InstructionDataset(val_data, tokenizer, device)
    
    print("Build collate_fn with partial to include pad_token_id, max_length, ignore_index, and device...")
    custome_collate = partial(
        collate_fn,
        pad_token_id=tokenizer.vocab.get('<|pad|>', 0),
        max_length=config.max_length,
        ignore_index=-100,
        device=device
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
        shuffle=not config.shuffle, # we don shuffle test data
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        collate_fn=custome_collate,
        shuffle=not config.shuffle, # we don shuffle validation data
        num_workers=config.num_workers,
    )
    
    return train_loader, test_loader, val_loader

if __name__ == "__main__":
    config = DataLoaderConfig.config('sftmodel/config.yaml')
    tokenizer = AlmondTokenizerGPT('basemodel/config.yaml')
    train_loader, test_loader, val_loader = create_dataloaders(config, tokenizer, collate_fn, DEVICE)
    print(f"DataLoaders created successfully. Train batches: {len(train_loader)}, Test batches: {len(test_loader)}, Validation batches: {len(val_loader)} >.<")