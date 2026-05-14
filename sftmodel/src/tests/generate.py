'''
src/tests/generate.py

Generate next-token prediction for testing sftmodel after fine-tuning >.<
'''

import torch
import os
import time

from basemodel.src.model.gpt import AlmondGPTModel
from basemodel.src.tokenizer.bpe import AlmondTokenizerGPT
from utils.common import load_model, load_yaml

# -------------------------------
BASEMODEL_CONFIG_PATH = 'basemodel/config.yaml'
SFTMODEL_CONFIG_PATH = 'sftmodel/config.yaml'

BASEMODEL_CONFIG = load_yaml(BASEMODEL_CONFIG_PATH)
SFTMODEL_CONFIG = load_yaml(SFTMODEL_CONFIG_PATH)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
FINETUNED_MODEL_PATH = 'sftmodel/model/finetuned/finetuned_model.pt'
# -------------------------------

def load_model_and_tokenizer():
    model = AlmondGPTModel(config_path=BASEMODEL_CONFIG_PATH).to(device=DEVICE)
    model = load_model(model=model, file_path=FINETUNED_MODEL_PATH, device=DEVICE)
    model.eval()
    
    tokenizer = AlmondTokenizerGPT(config_path=BASEMODEL_CONFIG_PATH)
    tokenizer.load(
        vocab_path=BASEMODEL_CONFIG['tokenizer']['vocab_path'] + 'vocab.json',
        merges_path=BASEMODEL_CONFIG['tokenizer']['merges_path'] + 'merges.json',
    )
    return model, tokenizer

def format_input(prompt: str):
    return f"<|startoftext|><|user|>\n{prompt}\n<|assistant|>\n"

def generate_test(model: AlmondGPTModel, tokenizer: AlmondTokenizerGPT, prompt: str, max_new_tokens: int = 200):
    ids = tokenizer.encode(format_input(prompt))
    idx = torch.tensor([ids], dtype=torch.long).to(DEVICE)
    
    with torch.no_grad():
        output = model.generate(idx, max_new_tokens=max_new_tokens)
    
    return tokenizer.decode(output[0].tolist())

def generate_stream(model: AlmondGPTModel, tokenizer: AlmondTokenizerGPT, prompt: str, max_new_tokens: int = 200):
    ids = tokenizer.encode(format_input(prompt))
    idx = torch.tensor([ids], dtype=torch.long).to(DEVICE)
    
    eos_id = tokenizer.SPECIAL_TOKEN.index('<|endoftext|>')

    print(prompt)
    print("[Answer]:")
    
    prompt_len = 0
    generated_text = ""
    
    with torch.no_grad():
        for token_id in model.generate_stream(idx, max_new_tokens=max_new_tokens):
            if token_id == eos_id:
                print("\n[End of generation]")
                break
            
            token_bytes = tokenizer.vocab.get(token_id, b'')
            try:
                token_str = token_bytes.decode('utf-8', errors='replace')
            except:
                token_str = '?'
            
            generated_text += token_str
            
            if "<|endoftext|>" in generated_text:
                generated_text = generated_text.split("<|endoftext|>")[0]
                print(generated_text[prompt_len:], end="", flush=True)
                print("\n[End of generation]")
                break
            
            print(token_str, end='', flush=True)
            prompt_len = len(generated_text)
            time.sleep(0.05)

if __name__ == "__main__":
    print("Loading fine-tuned model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer()
    
    QUESTION = [
        "Tell a story about a cherry tree.",
        "Write a story about Lily the princess.",
        "Write a story about a lost toy.",
        "Write a story about a dependable car.",
        "Tell me a story about a girl and a needle.",
        # "Tell a story about a big octopus.",
        # "Write a story about Tom the cat.",
        # "Tell me a story about a polite crab.",
        # "Write a story about a broken pole.",
    ]
    
    while True:
        prompt = input("Enter a prompt (or 'exit' to quit): ")
        if prompt.lower() == 'exit':
            print("Exiting...")
            break
        
        if prompt.lower() == 'go':
            for question in QUESTION:
                print("\n[Generated response]:")
                generate_stream(model, tokenizer, question)
            print()
        
        if prompt.lower() == 'gg':
            print()
            print("[Generated response]:")
            print(prompt)
            print("[Answer]:")
            print("You're welcome, i can like this cos my dev is Adin Ramdan Farelino >.<")
            print("[End of generation]")