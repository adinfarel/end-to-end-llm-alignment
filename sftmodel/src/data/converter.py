'''
src/data/converter.py

Convert raw .txt to .jsonl for easy loading in training >.<
'''

import json
import os
import random

# --------------------------------
RAW_DATA_INSTRUCTION_PATH = r'sftmodel\data\raw\instructions.txt'
RAW_DATA_OUTPUT_PATH = r'sftmodel\data\raw\instructions.jsonl'
# --------------------------------

def convert_txt_to_jsonl(input_path: str, output_path: str) -> None:
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = f.read()
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {e}")
    
    samples = data.split('<|startoftext|>')
    samples = [s.strip() for s in samples if s.strip()]
    
    if not os.path.exists(os.path.dirname(output_path)):
        print(f"Output directory does not exist. Creating directory: {os.path.dirname(output_path)}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in samples:
                text = "<|startoftext|>" + sample
                json_line = {"text": text}
                f.write(json.dumps(json_line) + '\n')
    except Exception as e:
        raise e

    print(f"Conversion completed. {len(samples)} samples written to {output_path}.")

def load_txt_to_list(input_path: str) -> list:
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = f.read()
        
        data_list = data.split('\n\n')
        data_list = [d.strip() for d in data_list if d.strip()]
        print(f"Loaded {len(data_list)} samples from {input_path}.")
        return data_list
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {e}")

def split_data(data_list: list, train_ratio: float = 0.8, test_ratio: float = 0.1, seed: int = 42) -> tuple:
    '''Split data into train, test, and validation sets >.<
    
    Args:
        data_list (list): List of data samples.
        train_ratio (float): Proportion of data to be used for training. The rest will be split between test and validation.
    
    Returns:
        tuple: (train_data, test_data, val_data)
    '''
    assert train_ratio + test_ratio < 1.0, "Train ratio and test ratio must sum to less than 1.0"
    data_list = data_list.copy()
    random.seed(seed)
    random.shuffle(data_list)
    
    total_samples = len(data_list)
    train_portion = int(total_samples * train_ratio)
    test_portion = int(total_samples * test_ratio)
    
    train_data = data_list[:train_portion]
    test_data = data_list[train_portion:train_portion + test_portion]
    val_data = data_list[train_portion + test_portion:]
    
    return train_data, test_data, val_data

if __name__ == "__main__":
    # convert_txt_to_jsonl(RAW_DATA_INSTRUCTION_PATH, RAW_DATA_OUTPUT_PATH)
    data_list = load_txt_to_list(RAW_DATA_INSTRUCTION_PATH)
    print(f"First sample:\n{data_list[:5]}")