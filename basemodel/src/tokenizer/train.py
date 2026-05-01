'''
src/tokenizer/train.py

Train BPE to get vocab and merges >.<
'''

from utils.common import load_yaml
from basemodel.src.tokenizer.bpe import AlmondTokenizerGPT

# -------------------------------
YAML_PATH = "basemodel/config.yaml"
CONFIG = load_yaml(YAML_PATH)
# -------------------------------

def train_tokenizer(raw_data_path: str, vocab_path: str, merges_path: str):
    try:
        with open(raw_data_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        print(f"Raw corpus text load from {raw_data_path}. Ready for training...")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {e}")

    tokenizer = AlmondTokenizerGPT(YAML_PATH)
    print(f'Tokenizer with vocab size: {tokenizer.vocab_size}')
    
    print(f"Training BPE starting..., Please wait for a minute...")
    tokenizer.train(text)
    
    print(f"Training done. Save vocabulary and merges.")
    tokenizer.save(vocab_path, merges_path)
    
    print('Tokenizer BPE Completed. All process done.')

if __name__ == "__main__":
    raw_data_path = CONFIG['dataset']['raw_save_path'] + 'tinystories.txt'
    vocab_path = CONFIG['tokenizer']['vocab_path'] + 'vocab.json'
    merges_path = CONFIG['tokenizer']['merges_path'] + 'merges.json'
    
    train_tokenizer(raw_data_path, vocab_path, merges_path)