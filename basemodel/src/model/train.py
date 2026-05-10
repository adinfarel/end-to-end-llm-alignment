'''
src/model/train.py

Train model, piece by piece
'''

import torch
import os
from dataclasses import dataclass

from basemodel.src.model.gpt import AlmondGPTModel
from basemodel.src.tokenizer.bpe import AlmondTokenizerGPT
from basemodel.src.model.utils import get_batch, eval_loss
from utils.common import load_yaml, load_bin, save_model

# ------------------------------
YAML_PATH = 'basemodel/config.yaml'
CONFIG = load_yaml(YAML_PATH)
PROCESSED_DATA_PATH = CONFIG['dataset']['processed_save_path'] + 'corpus.bin'
VOCAB_PATH = CONFIG['tokenizer']['vocab_path'] + 'vocab.json'
MERGES_PATH = CONFIG['tokenizer']['merges_path'] + 'merges.json'
MODEL_SAVE_DIR = CONFIG['models']['training']['MODEL_SAVE_DIR']
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
# ------------------------------

@dataclass
class TrainConfig:
    learning_rate: float
    MODEL_SAVE_DIR: str
    training: bool
    max_iters: int
    eval_interval: int
    eval_iters: int
    early_stopping: bool
    early_stopping_patience: int
    
    @classmethod
    def config(cls, yaml_path: str):
        cfg = load_yaml(yaml_path)['models']
        return cls(
            learning_rate=cfg['training']['learning_rate'],
            MODEL_SAVE_DIR=cfg['training']['model_save_dir'],
            training=cfg['training']['training'],
            max_iters=cfg['eval']['max_iters'],
            eval_interval=cfg['eval']['eval_interval'],
            eval_iters=cfg['eval']['eval_iters'],
            early_stopping=cfg['training']['early_stopping'],
            early_stopping_patience=cfg['training']['early_stopping_patience'],
        )

def main(config: TrainConfig, early_stopping: bool = True):
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
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
    
    best_val_loss = float('inf')
    scaler = torch.cuda.amp.GradScaler(enabled=(DTYPE == torch.float16))
    early_stopping_counter = 0
    early_stopping_patience = config.early_stopping_patience
    for iter in range(config.max_iters):
        if iter % config.eval_interval == 0:
            losses = eval_loss(model=model, data=data, config=CONFIG['models'], device=DEVICE)
            print(f"Step {iter} | Eval Loss: {losses:.4f}")
            
            if losses < best_val_loss:
                early_stopping_counter = 0
                best_val_loss = losses
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'iter': iter,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                save_model(model=checkpoint, file_path=os.path.join(MODEL_SAVE_DIR, 'best_model.pt'))
                print(f"--> New best model saved at step {iter} with loss {best_val_loss:.4f}")
            else:
                early_stopping_counter += 1
                if early_stopping and early_stopping_counter >= early_stopping_patience:
                    print(f"--> Early stopping at step {iter}")
                    break
        
        xb, yb = get_batch(data=data, batch_size=CONFIG['models']['training']['batch_size'], block_size=CONFIG['models']['training']['block_size'])
        xb, yb = xb.to(device=DEVICE), yb.to(device=DEVICE)
        
        with torch.autocast(device_type=DEVICE, dtype=DTYPE, enabled=(DEVICE == 'cuda')):
            logits, loss = model(xb, yb, use_cache=False)
            
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
    
    save_model(model={
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'iter': config.max_iters,
    }, file_path=os.path.join(MODEL_SAVE_DIR, 'ckpt_latest.pt'))
    print(f"Training done. Model save at {MODEL_SAVE_DIR}")

if __name__ == "__main__":
    config = TrainConfig.config(YAML_PATH)
    main(config, early_stopping=config.early_stopping)