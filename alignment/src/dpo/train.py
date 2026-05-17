'''
src/dpo/train.py

Train model with alignment DPO
'''

import os
import torch
from dataclasses import dataclass
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.common import load_yaml, load_model, save_model
from basemodel.src.tokenizer.bpe import AlmondTokenizerGPT
from basemodel.src.model.gpt import AlmondGPTModel
from alignment.src.data.loader import DPODatasetConfig, create_dpo_dataloaders
from alignment.src.dpo.dpo import DPOLoss

# -----------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CONFIG_PATH = 'alignment/config.yaml'
DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
# -----------------------------

@dataclass
class DPOTrainConfig:
    num_epochs: int
    learning_rate: float
    tokenizer_path: str
    models_path: str
    config_models_path: str
    save_model_dpo_path: str
    
    @classmethod
    def config(cls, yaml_path: str):
        cfg = load_yaml(yaml_path)
        return cls(
            num_epochs=cfg['alignment']['num_epochs'],
            learning_rate=cfg['alignment']['learning_rate'],
            tokenizer_path=cfg['tokenizer']['tokenizer_path'],
            models_path=cfg['models']['models_path'],
            config_models_path=cfg['models']['config_models'],
            save_model_dpo_path=cfg['alignment']['save_dpo_models']
        )

@torch.no_grad()
def eval_loss(pi_model: AlmondGPTModel, ref_model: AlmondGPTModel, val_loader: DataLoader):
    '''Calculate eval loss'''
    dpo_loss = DPOLoss(beta=0.1)
    pi_model.eval(); ref_model.eval()
    losses = []
    for data in val_loader:
        chosen_input_ids = data['chosen_input_ids']
        rejected_input_ids = data['rejected_input_ids']
        chosen_labels = data["chosen_labels"]
        rejected_labels = data["rejected_labels"]
        
        chosen_input_ids = chosen_input_ids.to(DEVICE); rejected_input_ids = rejected_input_ids.to(DEVICE)
        chosen_labels = chosen_labels.to(DEVICE); rejected_labels = rejected_labels.to(DEVICE)
        
        with torch.cuda.amp.autocast(enabled=True, dtype=DTYPE):
            pi_logits_chosen, _ = pi_model(chosen_input_ids, use_cache=False)
            ref_logits_chosen, _ = ref_model(chosen_input_ids, use_cache=False)
            pi_logits_rejected, _ = pi_model(rejected_input_ids,  use_cache=False)
            ref_logits_rejected, _ = ref_model(rejected_input_ids,  use_cache=False)
        
        # CALCULATE LOSS WITH LOG-PROBABILITIES + KL-DIVERGENCE
        loss = dpo_loss(
            pi_logits_chosen,
            pi_logits_rejected,
            ref_logits_chosen,
            ref_logits_rejected,
            chosen_labels,
            rejected_labels
        )
        
        losses.append(loss.item())
        
    pi_model.train()
    return sum(losses) / len(losses)

def train_dpo(config: DPOTrainConfig):
    '''Pipeline training Direct Preference Optimization (DPO) here >.<'''
    
    
    print("Load all component e.g. Models, Tokenizer, etc.")
    MODEL_PATH = os.path.join(config.models_path, 'finetuned_model.pt')
    MERGES_PATH = os.path.join(config.tokenizer_path, 'merges.json')
    VOCAB_PATH = os.path.join(config.tokenizer_path, 'vocab.json')
    
    pi_model = AlmondGPTModel(config_path=config.config_models_path)
    ref_model = AlmondGPTModel(config_path=config.config_models_path)
    tokenizer = AlmondTokenizerGPT(config_path=config.config_models_path)
    dpo_loss = DPOLoss(beta=0.1)
    
    pi_model = load_model(pi_model, MODEL_PATH, device=DEVICE)
    ref_model = load_model(ref_model, MODEL_PATH, device=DEVICE)
    pi_model = pi_model.to(DEVICE)
    ref_model = ref_model.to(DEVICE)
    
    tokenizer.load(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    
    optimizer = torch.optim.AdamW(pi_model.parameters(), lr=float(config.learning_rate))
    scaler = torch.cuda.amp.GradScaler(enabled=(DTYPE == torch.float16))
    
    # FREEZE REF MODEL
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    # TRAIN PI MODEL
    pi_model.train()
    
    print('Load datasets')
    config_dpo_dataset = DPODatasetConfig.config(yaml_path=CONFIG_PATH)
    train_loader, _, val_loader = create_dpo_dataloaders(config=config_dpo_dataset, tokenizer=tokenizer)
    
    # TRAINING LOOP
    print('Starting training loop...')
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        for data in progress_bar:
            optimizer.zero_grad(set_to_none=True)
            
            chosen_input_ids = data['chosen_input_ids']
            rejected_input_ids = data['rejected_input_ids']
            chosen_labels = data["chosen_labels"]
            rejected_labels = data["rejected_labels"]
            
            chosen_input_ids = chosen_input_ids.to(DEVICE); rejected_input_ids = rejected_input_ids.to(DEVICE)
            chosen_labels = chosen_labels.to(DEVICE); rejected_labels = rejected_labels.to(DEVICE)
        
            all_input_ids = torch.cat([chosen_input_ids, rejected_input_ids], dim=0)        
            with torch.cuda.amp.autocast(enabled=True, dtype=DTYPE):
                # pi_logits_chosen, _ = pi_model(chosen_input_ids, use_cache=False)
                # ref_logits_chosen, _ = ref_model(chosen_input_ids, use_cache=False)
                # pi_logits_rejected, _ = pi_model(rejected_input_ids,  use_cache=False)
                # ref_logits_rejected, _ = ref_model(rejected_input_ids,  use_cache=False)
                all_pi_logits, _ = pi_model(all_input_ids, use_cache=False)
                
                with torch.no_grad():
                    all_ref_logits, _ = ref_model(all_input_ids, use_cache=False)
                
                pi_logits_chosen, pi_logits_rejected = all_pi_logits.chunk(2, dim=0)
                ref_logits_chosen, ref_logits_rejected = all_ref_logits.chunk(2, dim=0)
            
            loss = dpo_loss(
                pi_logits_chosen,
                pi_logits_rejected,
                ref_logits_chosen,
                ref_logits_rejected,
                chosen_labels,
                rejected_labels
            )
            
            # Scale to prevent overflow vanishing gradient
            scaler.scale(loss).backward()
            
            # Unscale while update weight into real grad number
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(pi_model.parameters(), max_norm=1.0)
            
            # Update weight only Policy Model
            scaler.step(optimizer)
            scaler.update()
            
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        val_loss = eval_loss(
            pi_model=pi_model,
            ref_model=ref_model,
            val_loader=val_loader
        )
        print(f"Epoch {epoch+1}/{config.num_epochs} - Validation Loss: {val_loss:.4f}")
    
    print('Training loop complete')
    
    os.makedirs(os.path.dirname(config.save_model_dpo_path), exist_ok=True)
    checkpoint = {
        'model': pi_model.state_dict(),
        'epoch': config.num_epochs,
    }
    save_model(checkpoint=checkpoint, file_path=config.save_model_dpo_path)
    print(f'Model with DPO Alignment successfully save in {config.save_model_dpo_path}.')
    
if __name__ == "__main__":
    config = DPOTrainConfig.config(yaml_path=CONFIG_PATH)
    train_dpo(config=config)