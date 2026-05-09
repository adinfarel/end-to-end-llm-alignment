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
from utils.common import load_yaml, load_bin

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
    
    @classmethod
    def config(cls, yaml_path: str):
        cfg = load_yaml(yaml_path)['models']
        return cls(
            learning_rate=cfg['training']['learning_rate'],
            MODEL_SAVE_DIR=cfg['training']['MODEL_SAVE_DIR'],
            training=cfg['training']['training'],
            max_iters=cfg['eval']['max_iters'],
            eval_interval=cfg['eval']['eval_interval'],
            eval_iters=cfg['eval']['eval_iters'],
        )

def main(config: TrainConfig):
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
    
    for iter in range(config.max_iters):
        if iter % config.eval_interval == 0:
            losses = eval_loss(model=model, data=data, config=CONFIG['models'], device=DEVICE)
            print(f"Step {iter} | Eval Loss: {losses:.4f}")
            
            if losses < best_val_loss:
                best_val_loss = losses
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'iter': iter,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                torch.save(checkpoint, os.path.join(MODEL_SAVE_DIR, 'best_model.pt'))
                print(f"--> New best model saved at step {iter} with loss {best_val_loss:.4f}")
        
        xb, yb = get_batch(data=data, batch_size=CONFIG['models']['training']['batch_size'], block_size=CONFIG['models']['training']['block_size'])
        xb, yb = xb.to(device=DEVICE), yb.to(device=DEVICE)
        
        with torch.autocast(device_type='cuda', dtype=DTYPE):
            logits, loss = model(xb, yb, use_cache=False)
            
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
    
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'iter': config.max_iters,
    }, os.path.join(MODEL_SAVE_DIR, 'ckpt_latest.pt'))
    print(f"Training done. Model save at {MODEL_SAVE_DIR}")

if __name__ == "__main__":
    config = TrainConfig.config(YAML_PATH)
    main(config)