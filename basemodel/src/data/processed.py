'''
src/data/processed.py

Processing data text to token ID (numeric) in binary data
'''

from utils.common import save_bin, load_yaml
from basemodel.src.tokenizer.bpe import AlmondTokenizerGPT
from tqdm import tqdm

# --------------------------------
YAML_PATH = "basemodel/config.yaml"
CONFIG = load_yaml(YAML_PATH)
# --------------------------------

def processed_corpus(raw_data_path: str, processed_data_path: str, vocab_path: str, merges_path: str):
    print(f"Initialized tokenizer...")
    tokenizer = AlmondTokenizerGPT(YAML_PATH)
    try:
        tokenizer.load(
            vocab_path=vocab_path,
            merges_path=merges_path,
        )
        print(f"Vocabulary and merges has been loaded.")
    except Exception as e:
        raise e
    
    try:
        with open(raw_data_path, 'r', encoding='utf-8') as f:
            text = f.read()
            
        print(f"Raw corpus text {text[:50]}\nload from {raw_data_path} with len {len(text)}")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {e}")
    
    try:
        print(f"Process tokenizing text into binary starting...")
        lines = text.splitlines()
        all_tokens_ids = []
        extend = all_tokens_ids.extend
        print(f"Tokenizing {len(lines)} lines of text...")
        for line in tqdm(lines, desc="Processing token to token IDs..."):
            token_ids = tokenizer.encode(line)
            extend(token_ids)
        print(f"Corpus text completed successfully.")
    except Exception as e:
        raise e
    
    save_bin(processed_data_path, all_tokens_ids)
    print(f"Processed binary data save at path: {processed_data_path}")

if __name__ == "__main__":
    raw_data_path = CONFIG['dataset']['raw_save_path'] + 'tinystories.txt'
    processed_data_path = CONFIG['dataset']['processed_save_path'] + 'corpus.bin'
    vocab_path = CONFIG['tokenizer']['vocab_path'] + 'vocab.json'
    merges_path = CONFIG['tokenizer']['merges_path'] + 'merges.json'
    
    processed_corpus(
        raw_data_path,
        processed_data_path,
        vocab_path,
        merges_path
    )