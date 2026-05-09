'''
src/model/train.py

Train model, piece by piece
'''

import torch
from dataclasses import dataclass

from basemodel.src.model.gpt import AlmondGPTModel
from basemodel.src.tokenizer.bpe import AlmondTokenizerGPT
from basemodel.src.model.utils import get_batch, eval_loss
from utils.common import load_yaml, load_bin

# ------------------------------
YAML_PATH = 'basemodel/config.yaml'
CONFIG = load_yaml(YAML_PATH)
PROCESSED_DATA_PATH = CONFIG['dataset']['processed_save_path'] + 'corpus.bin'
VOCAB_PATH = CONFIG['tokenizer']['vocab_path'] + 'vocab.json'
MERGES_PATH = CONFIG['tokenizer']['merges_path'] + 'merges.json'
MODEL_SAVE_PATH = CONFIG['models']['training']['model_save_path'] + 'gpt_1.pt'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# ------------------------------

@dataclass
class TrainConfig:
    learning_rate: float
    model_save_path: str
    training: bool
    max_iters: int
    eval_interval: int
    eval_iters: int
    
    @classmethod
    def config(cls, yaml_path: str):
        cfg = load_yaml(yaml_path)['models']
        return cls(
            learning_rate=cfg['training']['learning_rate'],
            model_save_path=cfg['training']['model_save_path'],
            training=cfg['training']['training'],
            max_iters=cfg['eval']['max_iters'],
            eval_interval=cfg['eval']['eval_interval'],
            eval_iters=cfg['eval']['eval_iters'],
        )

def main(config: TrainConfig):
    # Initialize model and tokenizer
    tokenizer = AlmondTokenizerGPT(config_path=YAML_PATH)
    model = AlmondGPTModel(config_path=YAML_PATH).to(device=DEVICE)
    model.train()
    
    # Load vocab and data
    data = load_bin(PROCESSED_DATA_PATH)
    tokenizer.load(vocab_path=VOCAB_PATH, merges_path=MERGES_PATH)
    model.vocab_size = tokenizer.get_vocab_size
    
    # Training
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    for iter in range(config.max_iters):
        if iter % config.eval_interval == 0:
            losses = eval_loss(model=model, data=data, config=CONFIG['models'], device=DEVICE)
            print(f"Step {iter} | Eval Loss: {losses}")
        
        xb, yb = get_batch(data=data, batch_size=CONFIG['models']['training']['batch_size'], block_size=CONFIG['models']['training']['block_size'])
        
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Training done. Model save at {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    config = TrainConfig.config(YAML_PATH)
    main(config)