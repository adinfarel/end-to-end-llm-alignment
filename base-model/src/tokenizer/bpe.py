'''
data/tokenizer/bpe.py

Tokenize text to numeric based on vocab with algorithmic BPE
'''

import json
from utils.common import load_yaml, save_json, load_json
from tqdm import tqdm
from typing import List, Tuple, Dict

class AlmondTokenizerGPT:
    
    SPECIAL_TOKEN = [
        '<|endoftext|>',
        '<|pad|>',
        '<|user|>',
        "<|user|>",
        "<|assistant|>",
        "<|startoftext|>"
    ]
    
    def __init__(self, config_path: str) -> None:
        self.config = load_yaml(config_path)['tokenizer']
        self.vocab_size = self.config['vocab_size']
        self.single_byte_size = 256
        self.vocab = {}
        self.merges = {}
    
    def train(self, text,):
        print(f"TRAIN TEXT TO GET VOCAB...")
        
        tokens = text.encode('utf-8')
        print(f"Encode text to token with encoding=utf-8, output: {tokens[:100]}")
        num_merges = self.vocab_size - self.single_byte_size - len(self.SPECIAL_TOKEN)
        id_used = self.single_byte_size + len(self.SPECIAL_TOKEN)
        ids = tokens
        
        for i in tqdm(range(1, num_merges), desc="Process to get vocab id every word (BPE)..."):
            stats = self.get_stats(ids)
            pair = max(stats, key=stats.get)
            idx = id_used + i
            print(f"Merging pair: {pair} as new token ID: {idx}")
            self.merges[pair] = idx
            ids = self.merge_pair(pair, ids, idx)
        print(f"Training completed, total merges {len(self.merges)}")
            
    def get_stats(self, ids: List[int]) -> Dict:
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts
    
    def merge_pair(self, pair: Tuple[int, int], ids: List[int], idx: int) -> List[int]:
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i+1]) == pair:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def encode(self, text: str) -> List[int]:
        ids = list(text.encode('utf-8'))
        while len(ids) >= 2:
            stats = self.get_stats(ids)
            best_pair = min(stats, key=lambda pair: self.merges.get(pair, float('inf')))
            if best_pair not in self.merges:
                break
            ids = self.merge_pair(best_pair, ids, self.merges[best_pair])
        print(f"Encode text to token successfully: {ids[:10]}")
        return ids
    
    def decode(self, ids: List[int]) -> str:
        tokens = b''.join([self.vocab[id] for id in ids])
        text = tokens.decode('utf-8', errors='replace')
        print(f"Decode token to text succesfully: {text[:10]}")
        return text
    
    def get_vocab(self) -> Dict:
        self.vocab = {i: bytes([i]) for i in range(self.single_byte_size)}
        for i, token in enumerate(self.SPECIAL_TOKEN):
            self.vocab[self.single_byte_size + i] = token
        
        for pair, idx in self.merges.items():
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
        
        return self.vocab
    
    @property
    def get_merges(self) -> Dict:
        return self.get_merges
    
    def save(self, vocab_path: str, merges_path: str) -> None:
        try:
            vocab_data = {str(k): v.decode('latin-1') for k, v in self.vocab.items()}
            merges_data = {f"{k[0]},{k[1]}": v for k, v in self.merges.items()}
            save_json(vocab_path, vocab_data)
            save_json(merges_path, merges_data)
            print(f"Vocabulary saved to {vocab_path} and merges saved to {merges_path} >.<")
        except Exception as e:
            raise e
    
    def load(self, vocab_path: str, merges_path: str) -> None:
        try:
            vocab_data = load_json(vocab_path)
            merges_data = load_json(merges_path)
            self.vocab = {int(k): v.encode('latin-1') for k, v in vocab_data.items()}
            self.merges = {tuple(map(int, k.split(','))): v for k, v in merges_data.items()}
            print(f"Vocabulary loaded from {vocab_path} and merges loaded from {merges_path} >.<")
        except Exception as e:
            print(f"Do train first (tokenizer.train(...)): {e}")