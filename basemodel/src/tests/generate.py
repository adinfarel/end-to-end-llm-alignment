'''
src/tests/generate.py

Generate next-token prediction for testing the model
'''

import torch
import time
import os
from basemodel.src.model.gpt import AlmondGPTModel
from basemodel.src.tokenizer.bpe import AlmondTokenizerGPT
from utils.common import load_model, load_yaml

# -------------------------------
YAML_PATH = 'basemodel/config.yaml'
CONFIG = load_yaml(YAML_PATH)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# -------------------------------

def load_model_and_tokenizer():
    model = AlmondGPTModel(config_path=YAML_PATH).to(device=DEVICE)
    model = load_model(model=model, file_path=os.path.join(CONFIG['models']['training']['model_save_dir'], 'best_model.pt'), device=DEVICE)
    model.eval()
    
    tokenizer = AlmondTokenizerGPT(config_path=YAML_PATH)
    tokenizer.load(
        vocab_path=CONFIG['tokenizer']['vocab_path'] + 'vocab.json',
        merges_path=CONFIG['tokenizer']['merges_path'] + 'merges.json',
    )
    
    return model, tokenizer
    
def generate_test(model: AlmondGPTModel, tokenizer: AlmondTokenizerGPT, prompt: str, max_new_tokens: int = 1000):
    '''Inference model for next-token prediction'''
    
    ids = tokenizer.encode(prompt)
    idx = torch.tensor([ids], dtype=torch.long).to(DEVICE)
    
    with torch.no_grad():
        output = model.generate(idx, max_new_tokens=max_new_tokens)
    
    return tokenizer.decode(output[0].tolist())

def generate_stream(model: AlmondGPTModel, tokenizer: AlmondTokenizerGPT, prompt: str, max_new_tokens: int = 1000):
    '''Inference model for next-token prediction with streaming output'''
    ids = tokenizer.encode(prompt)
    idx = torch.tensor([ids], dtype=torch.long).to(DEVICE)
    
    for token_id in model.generate_stream(idx, max_new_tokens=max_new_tokens):
        token_bytes = tokenizer.vocab.get(token_id, b'')
        try:
            token_str = token_bytes.decode('utf-8', errors='replace')
        except:
            token_str = '?'
        
        eos_id = tokenizer.SPECIAL_TOKEN.index('<|endoftext|>')
        
        if token_id == eos_id:
            print("\n[End of generation]")
            break
        
        print(token_str, end='', flush=True)
        time.sleep(0.05)

if __name__ == "__main__":
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer()
    
    while True:
        prompt = input("Enter a prompt (or 'exit' to quit): ")
        if prompt.lower() == 'exit':
            print("Exiting...")
            break
        
        result = generate_stream(model, tokenizer, prompt)
        print(f"Generated text:\n{result}\n")