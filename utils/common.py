'''
utils/common.py

Utility for common needs for code 
'''

import json
import yaml
import os
import torch
import numpy as np
from typing import Dict, Any

def save_json(file_path: str, data: Dict[str, Any]) -> None:
    if not os.path.exists(os.path.dirname(file_path)):
        print(f"Create directory for {file_path}.")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"File JSON succesfully saved.")

def load_json(file_path: str) -> Dict[str, Any]:
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        print(f"Successfully loaded at {file_path}.")
        return data
    
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {e}")

def save_yaml(file_path: str, data: Dict[str, Any]) -> None:
    if not os.path.exists(os.path.dirname(file_path)):
        print(f"File not found. Create directory: {file_path}")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w') as f:
        yaml.safe_dump(data, f)
    
    print(f"File YAML successfully saved.")

def load_yaml(file_path: str) -> Dict[str, Any]:
    try:
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        
        print(f"Data succesfully loaded at {file_path}.")
        return data
    
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {e}")

def save_bin(file_path: str, data: np.ndarray) -> None:
    if not os.path.exists(os.path.dirname(file_path)):
        print(f"Create directory for path: {file_path}")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    if isinstance(data, list):
        data = np.array(data, dtype=np.uint16)
    
    data.tofile(file_path)
    print(f"Binary file succesfully save in: {file_path}")

def load_bin(file_path: str) -> np.ndarray:
    try:
        data = np.memmap(file_path, dtype=np.uint16, mode='r')
        print(f"Binary data successfully loaded.")
        return data
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {e}")

def save_model(checkpoint: Dict, file_path: str) -> None:
    if not os.path.exists(os.path.dirname(file_path)):
        print(f"Create directory for path: {file_path}")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    torch.save(checkpoint, file_path)
    print(f"Model successfully saved in: {file_path}")

def load_model(model: torch.nn.Module, file_path: str, device: str) -> torch.nn.Module:
    try:
        checkpoint = torch.load(file_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model'])
        print(f"Model weights successfully loaded from: {file_path}")
        return model
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {e}")