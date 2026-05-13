'''
src/tokenizer/bpe.py

Tokenize text to numeric based on vocab with algorithmic BPE
'''

import json
import re
from utils.common import load_yaml, save_json, load_json
from tqdm import tqdm
from typing import List, Tuple, Dict
from collections import Counter

class AlmondTokenizerGPT:
    
    SPECIAL_TOKEN = [
        '<|endoftext|>',
        '<|pad|>',
        '<|user|>',
        "<|assistant|>",
        "<|startoftext|>"
    ]
    
    MAX_CHARS = 500_000
    
    def __init__(self, config_path: str) -> None:
        self.config = load_yaml(config_path)['tokenizer']
        self.vocab_size = self.config['vocab_size']
        self.single_byte_size = 256
        self.vocab = {}
        self.merges = {}
    
    def train(self, text,):
        print(f"TRAIN TEXT TO GET VOCAB...")
        
        tokens = text[:self.MAX_CHARS].encode('utf-8')
        print(f"Encode text to token with encoding=utf-8, output: {tokens[:100]}")
        num_merges = self.vocab_size - self.single_byte_size - len(self.SPECIAL_TOKEN)
        id_used = self.single_byte_size + len(self.SPECIAL_TOKEN)
        ids = list(tokens)
        
        for i in tqdm(range(0, num_merges), desc="Process to get vocab id every word (BPE)..."):
            stats = self.get_stats(ids)
            if not stats:
                break
            
            pair = max(stats, key=stats.get)
            freq = stats[pair]
            if freq < 5:
                print(f"Freq under 5, prefer stop cause it's rarely token.")
                break
            
            idx = id_used + i
            if i % 100 == 0:
                print(f"Step {i}: merging {pair} -> {idx}")
            self.merges[pair] = idx
            ids = self.merge_pair(pair, ids, idx)
        print(f"Training completed, total merges {len(self.merges)}")
        self.get_vocab()
            
    def get_stats(self, ids: List[int]) -> Dict:
        return Counter(zip(ids, ids[1:]))
    
    def merge_pair(self, pair: Tuple[int, int], ids: List[int], idx: int) -> List[int]:
        new_ids = []
        append = new_ids.append
        a, b = pair
        
        i = 0
        n = len(ids)
        while i < n:
            
            if i < n - 1 and ids[i] == a and ids[i+1] == b:
                append(idx)
                i += 2
            else:
                append(ids[i])
                i += 1
                
        return new_ids

    def encode(self, text: str) -> List[int]:
        pattern = '(' + '|'.join(re.escape(t) for t in self.SPECIAL_TOKEN) + ')'
        parts = re.split(pattern, text)
        
        ids = []
        for part in parts:
            if part in self.SPECIAL_TOKEN:
                special_idx = self.SPECIAL_TOKEN.index(part)
                vocab_id = self.single_byte_size + special_idx
                ids.append(vocab_id)

            else:
                if not part:
                    continue
                byte_ids = list(part.encode('utf-8'))
                while len(byte_ids) >= 2:
                    stats = self.get_stats(byte_ids)
                    best_pair = min(stats, key=lambda pair: self.merges.get(pair, float('inf')))
                    if best_pair not in self.merges:
                        break
                    byte_ids = self.merge_pair(best_pair, byte_ids, self.merges[best_pair])
                ids.extend(byte_ids)
        return ids
                    
        # ids = list(text.encode('utf-8'))
        # while len(ids) >= 2:
        #     stats = self.get_stats(ids)
        #     best_pair = min(stats, key=lambda pair: self.merges.get(pair, float('inf')))
        #     if best_pair not in self.merges:
        #         break
        #     ids = self.merge_pair(best_pair, ids, self.merges[best_pair])
        # print(f"Encode text to token successfully: {ids[:10]}")
        # return ids
    
    def decode(self, ids: List[int]) -> str:
        tokens = b''.join([self.vocab[id] for id in ids])
        text = tokens.decode('utf-8', errors='replace')
        print(f"Decode token to text succesfully: {text[:10]}")
        return text
    
    def get_vocab(self) -> Dict:
        self.vocab = {i: bytes([i]) for i in range(self.single_byte_size)}
        for i, token in enumerate(self.SPECIAL_TOKEN):
            self.vocab[self.single_byte_size + i] = token.encode('utf-8')
        
        for pair, idx in self.merges.items():
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
        
        return self.vocab
    
    @property
    def get_merges(self) -> Dict:
        return self.merges
    
    @property
    def get_vocab_size(self) -> int:
        return len(self.vocab)
    
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