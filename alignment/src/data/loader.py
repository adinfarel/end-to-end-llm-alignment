'''
src/data/loader.py

Plugin dataset into DataLoader 
'''

import os
import torch

from dataclasses import dataclass
from functools import partial
from torch.utils.data import DataLoader
from typing import Tuple

from basemodel.src.tokenizer.bpe import AlmondTokenizerGPT
from utils.common import load_yaml, load_json
from sftmodel.src.data.converter import split_data
from alignment.src.data.dataset import DpoDatasets, collate_fn

# ---------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ---------------------------
@dataclass
class DPODatasetConfig:
    batch_size: int
    max_length: int
    raw_data_path: str
    train_ratio: float
    test_ratio: float
    shuffle: bool
    num_workers: int
    model_sft_path: str
    tokenizer_path: str
    
    @classmethod
    def config(cls, yaml_path: str):
        cfg = load_yaml(yaml_path)
        return cls(
            batch_size=cfg['alignment']['batch_size'],
            max_length=cfg['alignment']['max_length'],
            raw_data_path=cfg['alignment']['dpo_dataset_path'],
            train_ratio=cfg['alignment']['train_ratio'],
            test_ratio=cfg['alignment']['test_ratio'],
            shuffle=cfg['alignment']['shuffle'],
            num_workers=cfg['alignment']['num_workers'],
            model_sft_path=cfg['models']['models_path'],
            tokenizer_path=cfg['tokenizer']['tokenizer_path']
        )

def create_dpo_dataloaders(
    config: DPODatasetConfig,
    tokenizer: AlmondTokenizerGPT
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    '''
    Get dataset into DataLoader
    
    Args:
        config: all config that needed for create dataloader
        tokenizer: for tokenize text to input ids
    
    Returns:
        train_loader: DataLoader for train
        test_loader: DataLoader for test
        val_loader: DataLoader for validation
    '''
    print("Starting create DPO Dataset into DataLoader...")
    
    print(f"Load json data from path {config.raw_data_path}...")
    data = load_json(config.raw_data_path)
    
    print(f"Split dataset into ratio: train={config.train_ratio}, test={config.test_ratio}, validation={1- config.train_ratio - config.test_ratio}")
    train_data, test_data, val_data = split_data(data_list=data, train_ratio=config.train_ratio, test_ratio=config.test_ratio)
    
    print(f"Initialized dataset into {DpoDatasets.__name__}")
    train_dataset = DpoDatasets(
        train_data,
        tokenizer
    )
    test_dataset = DpoDatasets(
        test_data,
        tokenizer
    )
    val_dataset = DpoDatasets(
        val_data,
        tokenizer
    )
    
    PAD_TOKEN_ID = tokenizer.single_byte_size + tokenizer.SPECIAL_TOKEN.index('<|pad|>')
    
    print("Create DataLoader for each split data")
    partial_collate_fn = partial( 
        collate_fn,
        pad_token_id=PAD_TOKEN_ID,
        max_length=config.max_length,
    ) # Cause DataLoader just send one parameters that is Batch of DpoDataset class that plugin into collate_fn (function that running DataLoader), 
    # we need partial to fill parameter unless batch parameter
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        collate_fn=partial_collate_fn,
        num_workers=config.num_workers
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=partial_collate_fn,
        num_workers=config.num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=partial_collate_fn,
        num_workers=config.num_workers
    )
    
    print(f"Pipeline for creating DataLoader complete. Yayyy >.<")
    return train_loader, test_loader, val_loader
