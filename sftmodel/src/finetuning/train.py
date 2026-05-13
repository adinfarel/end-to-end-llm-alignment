'''
src/finetuning/train.py

Train the model with fine-tuning >//<
'''

import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataclasses import dataclass

from basemodel.src.model.gpt import AlmondGPTModel
from basemodel.src.tokenizer.bpe import AlmondTokenizerGPT
from sftmodel.src.data.loader import create_dataloaders, DataLoaderConfig
from sftmodel.src.data.datasets import collate_fn
from utils.common import load_yaml, save_model, load_model

# --------------------------------
BASEMODEL_CONFIG = load_yaml('basemodel/config.yaml')
PRETRAIN_MODEL_PATH = BASEMODEL_CONFIG['models']['training']['model_save_dir'] + 'best_model.pt'
VOCAB_PATH = BASEMODEL_CONFIG['tokenizer']['vocab_path'] + 'vocab.json'
MERGES_PATH = BASEMODEL_CONFIG['tokenizer']['merges_path'] + 'merges.json'
SFTMODEL_CONFIG_PATH = 'sftmodel/config.yaml'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
# --------------------------------

@dataclass
class FineTuneConfig:
    num_epochs: int
    learning_rate: float
    model_finetune_path: str
    
    @classmethod
    def config(cls, yaml_path: str):
        cfg = load_yaml(yaml_path)
        return cls(
            num_epochs=cfg['finetune']['num_epochs'],
            learning_rate=cfg['finetune']['learning_rate'],
            model_finetune_path=cfg['finetune']['model_finetune_path']
        )

def eval_loss(model: AlmondGPTModel, val_loader: DataLoader):
    '''Evaluate the model on the validation set and return (1 iteration) loss'''
    model.eval()
    losses = []
    for inputs, targets in val_loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=DTYPE):
            _, loss = model(inputs, targets)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)

def train_finetune(
    model: AlmondGPTModel,
    tokenizer: AlmondTokenizerGPT,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    config: FineTuneConfig,
):
    '''Train the model with fine-tuning >//<'''
    model.to(DEVICE)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scaler = torch.cuda.amp.GradScaler(enabled=(DTYPE == torch.float16))
    epoch = 0
    
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        for inputs, targets in progress_bar:
            
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            with torch.cuda.amp.autocast(enabled=True, dtype=DTYPE):
                _, loss = model(inputs, targets, use_cache=False)
            
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        val_loss = eval_loss(model, val_loader)
        print(f"Epoch {epoch+1}/{config.num_epochs} - Validation Loss: {val_loss:.4f}")
    
    # Save the fine-tuned model
    os.makedirs(config.model_finetune_path, exist_ok=True)
    checkpoint = {
        'model': model.state_dict(),
        'epoch': config.num_epochs,
    }
    save_model(checkpoint, os.path.join(config.model_finetune_path, 'finetuned_model.pt'))
    print(f"Fine-tuned model saved to {config.model_finetune_path}")

if __name__ == "__main__":
    # Load configurations
    print("Loading configurations...")
    dataloader_cfg = DataLoaderConfig.config(SFTMODEL_CONFIG_PATH)
    finetune_cfg = FineTuneConfig.config(SFTMODEL_CONFIG_PATH)
    
    # Initialize tokenizer and model
    print("Initializing model and tokenizer...")
    tokenizer = AlmondTokenizerGPT(config_path=SFTMODEL_CONFIG_PATH)
    model = AlmondGPTModel(config_path=SFTMODEL_CONFIG_PATH).to(DEVICE)
    
    # Load vocab and state dict for fine-tuning
    print("Loading pre-trained model and tokenizer vocab...")
    model = load_model(model, PRETRAIN_MODEL_PATH, device=DEVICE)
    tokenizer.load(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, test_loader, val_loader = create_dataloaders(
        config=dataloader_cfg,
        tokenizer=tokenizer,
        collate_fn=collate_fn,
        device=DEVICE
    )
    print("Dataloaders created.")
    
    # Start fine-tuning
    print("Starting fine-tuning...")
    train_finetune(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
        config=finetune_cfg,
    )
    
    print("Fine-tuning completed.")