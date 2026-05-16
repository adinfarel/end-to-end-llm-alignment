'''
src/pipeline/main.py

This is the main file for the entire pipeline of training the tokenizer and the model. It will call the train.py files in both tokenizer and model directories to execute the training process.
'''
import os
import sys
from basemodel.src.tokenizer.train import train_tokenizer
from basemodel.src.data.load import load_datasets
from basemodel.src.data.processed import processed_corpus
from basemodel.src.model.train import main as train_model
from basemodel.src.model.train import TrainConfig
from utils.common import load_yaml

# -------------------------------
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
YAML_PATH = "basemodel/config.yaml"
CONFIG = load_yaml(YAML_PATH)
# -------------------------------

if __name__ == "__main__":
    # Load dataset from HuggingFace and save as .txt file
    load_datasets(yaml_path=YAML_PATH)
    
    # Train tokenizer to get vocab and merges
    raw_data_path = CONFIG['dataset']['raw_save_path'] + 'tinystories.txt'
    vocab_path = CONFIG['tokenizer']['vocab_path'] + 'vocab.json'
    merges_path = CONFIG['tokenizer']['merges_path'] + 'merges.json'
    train_tokenizer(raw_data_path, vocab_path, merges_path)
    CONFIG = load_yaml(YAML_PATH)  # Reload config to get updated vocab size
    
    # Process raw text data to token IDs and save as binary file
    processed_data_path = CONFIG['dataset']['processed_save_path'] + 'corpus.bin'
    processed_corpus(raw_data_path, processed_data_path, vocab_path, merges_path)
    
    # The main idea of this project >.<
    train_config = TrainConfig.config(yaml_path=YAML_PATH)
    train_model(config=train_config, early_stopping=train_config.early_stopping)