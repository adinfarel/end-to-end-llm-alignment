'''
data/raw/load.py 

Load dataset TinyStories from huggingface
'''

import os
from datasets import load_dataset
from utils.common import load_yaml
from dataclasses import dataclass
from tqdm import tqdm

@dataclass
class Datasets:
    name_dataset: str
    raw_save_path: str
    split: str
    
    @classmethod
    def config(cls, config_path: str):
        config = load_yaml(config_path)
        
        return cls(
            name_dataset=config['dataset'].get('name_dataset', ''),
            raw_save_path=config['dataset'].get('raw_save_path', ''),
            split=config['dataset'].get('split', '')
        )

def load_datasets(yaml_path):
    print("=============LOADED DATASET=============")
    config = Datasets.config(yaml_path=yaml_path)
    ds = load_dataset(config.name_dataset, split=config.split)
    
    print(f"Dataset successfully load from HuggingFace.")
    raw_save_path = config.raw_save_path + "tinystories.txt"
    if not os.path.exists(config.raw_save_path):
        print(f'Create directory raw save path: {config.raw_save_path}')
        os.makedirs(config.raw_save_path, exist_ok=True)
    
    with open(raw_save_path, 'a', encoding='utf-8') as f:
        for item in tqdm(ds, desc="Export dataset into .txt file"):
            f.write(item['text'].strip() + '<|endoftext|>\n')

    print(f"Dataset done export .txt file at {raw_save_path} with total {len(ds)} samples")

if __name__ == "__main__":
    load_datasets("base-model/config.yaml")