import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import yaml
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, asdict
import shap

# ============================================================================
# Configuration Management
# ============================================================================

@dataclass
class MLMConfig:
    """Configuration for MLM pre-training"""
    # Masking strategy
    mask_prob: float = 0.15
    mask_prob_ts: float = 0.15  # Specific for time series
    mask_prob_cat: float = 0.15  # Specific for categorical
    mask_prob_cont: float = 0.15  # Specific for continuous
    replace_prob: float = 0.8
    random_prob: float = 0.1
    
    # Training hyperparameters
    epochs: int = 50
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_epochs: int = 5
    
    # Loss weighting
    ts_loss_weight: float = 1.0
    cat_loss_weight: float = 1.0
    cont_loss_weight: float = 1.0
    contrastive_weight: float = 0.0  # Set > 0 to enable contrastive learning
    
    # Contrastive learning
    temperature: float = 0.07
    projection_dim: int = 128
    
    # Checkpointing
    save_best: bool = True
    checkpoint_dir: str = "./checkpoints"
    
    # Early stopping
    patience: int = 10
    min_delta: float = 1e-4
    
    # Validation
    val_frequency: int = 1  # Validate every N epochs
    
    @classmethod
    def from_yaml(cls, yaml_path: str):
        """Load configuration from YAML file"""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def to_yaml(self, yaml_path: str):
        """Save configuration to YAML file"""
        with open(yaml_path, 'w') as f:
            yaml.dump(asdict(self), f, default_flow_style=False)


# ============================================================================
# Enhanced MLM Model with Contrastive Learning
# ============================================================================

class TSTabFusionMLM(nn.Module):
    """
    Enhanced Masked Language Modeling wrapper with contrastive learning.
    """
    def __init__(self, backbone, config: MLMConfig):
        super().__init__()
        self.backbone = backbone
        self.config = config
        
        d_model = backbone.W_P.out_channels
        
        # Reconstruction heads
        self.ts_head = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.LayerNorm(d_model * 2),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, backbone.W_P.in_channels)
        )
        
        # Categorical reconstruction heads
        if backbone.n_emb != 0:
            n_classes = [emb.num_embeddings for emb in backbone.embeds]
            self.cat_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(d_model, d_model),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(d_model, n_class)
                ) for n_class in n_classes
            ])
        else:
            self.cat_heads = None
        
        # Continuous reconstruction head
        if backbone.n_cont != 0:
            self.cont_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(d_model, 1)
            )
        else:
            self.cont_head = None
        
        # Contrastive learning projection head
        if config.contrastive_weight > 0:
            self.projection_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, config.projection_dim)
            )
        else:
            self.projection_head = None
    
    def create_mlm_mask(self, shape, device, mask_prob):
        """Create random mask for MLM with specified probability."""
        mask = torch.rand(shape, device=device) < mask_prob
        return mask
    
    def mask_time_series(self, x_ts):
        """Mask time series data with improved strategy."""
        bs, c_in, seq_len = x_ts.shape
        device = x_ts.device
        
        # Create mask per timestep
        mask = self.create_mlm_mask((bs, seq_len), device, self.config.mask_prob_ts)
        
        original_x_ts = x_ts.clone()
        masked_x_ts = x_ts.clone()
        
        # Apply masking
        for i in range(bs):
            for t in range(seq_len):
                if mask[i, t]:
                    rand_val = torch.rand(1).item()
                    if rand_val < self.config.replace_prob:
                        masked_x_ts[i, :, t] = 0
                    elif rand_val < self.config.replace_prob + self.config.random_prob:
                        random_t = torch.randint(0, seq_len, (1,)).item()
                        masked_x_ts[i, :, t] = x_ts[i, :, random_t]
        
        return masked_x_ts, mask, original_x_ts
    
    def mask_categorical(self, x_cat):
        """Mask categorical features."""
        if x_cat is None or x_cat.shape[1] == 0:
            return x_cat, None, x_cat
        
        bs, n_cat = x_cat.shape
        device = x_cat.device
        
        mask = self.create_mlm_mask((bs, n_cat), device, self.config.mask_prob_cat)
        
        original_x_cat = x_cat.clone()
        masked_x_cat = x_cat.clone()
        
        for i in range(bs):
            for j in range(n_cat):
                if mask[i, j]:
                    rand_val = torch.rand(1).item()
                    n_classes = self.backbone.embeds[j].num_embeddings
                    if rand_val < self.config.replace_prob:
                        masked_x_cat[i, j] = 0
                    elif rand_val < self.config.replace_prob + self.config.random_prob:
                        masked_x_cat[i, j] = torch.randint(0, n_classes, (1,)).item()
        
        return masked_x_cat, mask, original_x_cat
    
    def mask_continuous(self, x_cont):
        """Mask continuous features with standardization-aware masking."""
        if x_cont is None or x_cont.shape[1] == 0:
            return x_cont, None, x_cont
        
        bs, n_cont = x_cont.shape
        device = x_cont.device
        
        mask = self.create_mlm_mask((bs, n_cont), device, self.config.mask_prob_cont)
        
        original_x_cont = x_cont.clone()
        masked_x_cont = x_cont.clone()
        
        # Use feature statistics for better masking
        feature_means = x_cont.mean(dim=0, keepdim=True)
        
        for i in range(bs):
            for j in range(n_cont):
                if mask[i, j]:
                    rand_val = torch.rand(1).item()
                    if rand_val < self.config.replace_prob:
                        # Use feature mean instead of 0
                        masked_x_cont[i, j] = feature_means[0, j]
                    elif rand_val < self.config.replace_prob + self.config.random_prob:
                        random_idx = torch.randint(0, bs, (1,)).item()
                        masked_x_cont[i, j] = x_cont[random_idx, j]
        
        return masked_x_cont, mask, original_x_cont
    
    def forward_encoder(self, x_ts, x_cat, x_cont):
        """Forward pass through encoder."""
        # Handle NaN values
        if hasattr(self.backbone, '_key_padding_mask'):
            x_ts, key_padding_mask = self.backbone._key_padding_mask(x_ts)
        else:
            mask = torch.isnan(x_ts)
            if mask.any():
                x_ts = x_ts.clone()
                x_ts[mask] = 0
            key_padding_mask = None
        
        # Time series encoding
        x = self.backbone.W_P(x_ts).transpose(1, 2)
        
        # Categorical encoding
        if self.backbone.n_emb != 0 and x_cat is not None:
            x_cat_emb = [e(x_cat[:, i]).unsqueeze(1) for i, e in enumerate(self.backbone.embeds)]
            x_cat_emb = torch.cat(x_cat_emb, 1)
            x = torch.cat([x, x_cat_emb], 1)
        
        # Continuous encoding
        if self.backbone.n_cont != 0 and x_cont is not None:
            x_cont_emb = self.backbone.conv(x_cont.unsqueeze(1)).transpose(1, 2)
            x = torch.cat([x, x_cont_emb], 1)
        
        # Positional encoding
        x += self.backbone.pos_enc
        
        if self.backbone.res_drop is not None:
            x = self.backbone.res_drop(x)
        
        # Transformer
        import inspect
        transformer_sig = inspect.signature(self.backbone.transformer.forward)
        if 'key_padding_mask' in transformer_sig.parameters:
            x = self.backbone.transformer(x, None, key_padding_mask)
        else:
            x = self.backbone.transformer(x, None)
        
        if key_padding_mask is not None:
            x = x * torch.logical_not(key_padding_mask.unsqueeze(1))
        
        return x
    
    def contrastive_loss(self, z1, z2):
        """Compute NT-Xent contrastive loss."""
        batch_size = z1.shape[0]
        
        # Normalize
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        # Compute similarity matrix
        representations = torch.cat([z1, z2], dim=0)
        similarity_matrix = F.cosine_similarity(
            representations.unsqueeze(1), 
            representations.unsqueeze(0), 
            dim=2
        )
        
        # Create labels
        labels = torch.arange(batch_size, device=z1.device)
        labels = torch.cat([labels + batch_size, labels], dim=0)
        
        # Mask out self-similarity
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z1.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, -float('inf'))
        
        # Compute loss
        similarity_matrix = similarity_matrix / self.config.temperature
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss
    
    def forward(self, x_ts, x_cat, x_cont, return_contrastive=False):
        """Forward pass with MLM and optional contrastive learning."""
        # Create two augmented views for contrastive learning
        if return_contrastive and self.config.contrastive_weight > 0:
            # First view
            masked_ts1, ts_mask1, original_ts = self.mask_time_series(x_ts)
            masked_cat1, cat_mask1, original_cat = self.mask_categorical(x_cat)
            masked_cont1, cont_mask1, original_cont = self.mask_continuous(x_cont)
            
            # Second view (different masking)
            masked_ts2, _, _ = self.mask_time_series(x_ts)
            masked_cat2, _, _ = self.mask_categorical(x_cat)
            masked_cont2, _, _ = self.mask_continuous(x_cont)
            
            encoder_output1 = self.forward_encoder(masked_ts1, masked_cat1, masked_cont1)
            encoder_output2 = self.forward_encoder(masked_ts2, masked_cat2, masked_cont2)
            
            # Use first view for reconstruction
            encoder_output = encoder_output1
            ts_mask, cat_mask, cont_mask = ts_mask1, cat_mask1, cont_mask1
            
            # Global pooling for contrastive loss
            z1 = encoder_output1.mean(dim=1)  # [bs, d_model]
            z2 = encoder_output2.mean(dim=1)
            
            # Project
            z1 = self.projection_head(z1)
            z2 = self.projection_head(z2)
            
            contrastive_loss = self.contrastive_loss(z1, z2)
        else:
            masked_ts, ts_mask, original_ts = self.mask_time_series(x_ts)
            masked_cat, cat_mask, original_cat = self.mask_categorical(x_cat)
            masked_cont, cont_mask, original_cont = self.mask_continuous(x_cont)
            
            encoder_output = self.forward_encoder(masked_ts, masked_cat, masked_cont)
            contrastive_loss = None
        
        seq_len = x_ts.shape[2]
        n_cat = x_cat.shape[1] if x_cat is not None else 0
        n_cont = x_cont.shape[1] if x_cont is not None else 0
        
        # Split encoder output
        ts_output = encoder_output[:, :seq_len, :]
        cat_output = encoder_output[:, seq_len:seq_len+n_cat, :] if n_cat > 0 else None
        cont_output = encoder_output[:, seq_len+n_cat:, :] if n_cont > 0 else None
        
        losses = {}
        
        # Time series reconstruction
        if ts_mask.any():
            ts_pred = self.ts_head(ts_output).transpose(1, 2)
            ts_loss = F.mse_loss(
                ts_pred[ts_mask.unsqueeze(1).expand_as(ts_pred)],
                original_ts[ts_mask.unsqueeze(1).expand_as(original_ts)]
            )
            losses['ts_loss'] = ts_loss * self.config.ts_loss_weight
        
        # Categorical reconstruction
        if cat_mask is not None and cat_mask.any():
            cat_losses = []
            for i in range(n_cat):
                if cat_mask[:, i].any():
                    cat_pred = self.cat_heads[i](cat_output[:, i, :])
                    cat_loss = F.cross_entropy(
                        cat_pred[cat_mask[:, i]],
                        original_cat[cat_mask[:, i], i]
                    )
                    cat_losses.append(cat_loss)
            if cat_losses:
                losses['cat_loss'] = torch.stack(cat_losses).mean() * self.config.cat_loss_weight
        
        # Continuous reconstruction
        if cont_mask is not None and cont_mask.any():
            cont_pred = self.cont_head(cont_output).squeeze(-1)
            cont_loss = F.mse_loss(
                cont_pred[cont_mask],
                original_cont[cont_mask]
            )
            losses['cont_loss'] = cont_loss * self.config.cont_loss_weight
        
        # Add contrastive loss
        if contrastive_loss is not None:
            losses['contrastive_loss'] = contrastive_loss * self.config.contrastive_weight
        
        # Total loss
        total_loss = sum(losses.values())
        losses['total_loss'] = total_loss
        
        return losses


# ============================================================================
# Training with All Enhancements
# ============================================================================

class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""
    def __init__(self, patience=10, min_delta=1e-4, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self.mode == 'min':
            if score > self.best_score - self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.counter = 0
        else:  # mode == 'max'
            if score < self.best_score + self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.counter = 0
        
        return self.early_stop


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """Create learning rate scheduler with warmup."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def pretrain_mlm_enhanced(
    model, 
    train_loader, 
    config: MLMConfig,
    val_loader=None,
    device='cuda'
):
    """
    Enhanced pre-training with all features.
    """
    from tqdm.auto import tqdm
    from collections import defaultdict
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.lr,
        weight_decay=config.weight_decay
    )
    
    # Learning rate scheduler
    num_training_steps = config.epochs * len(train_loader)
    num_warmup_steps = config.warmup_epochs * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps, num_training_steps
    )
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config.patience,
        min_delta=config.min_delta
    )
    
    # Checkpoint directory
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'ts_loss': [],
        'cat_loss': [],
        'cont_loss': [],
        'contrastive_loss': [],
        'lr': []
    }
    
    best_val_loss = float('inf')
    
    for epoch in range(config.epochs):
        # Training
        model.train()
        meters = defaultdict(lambda: {'sum': 0, 'count': 0})
        pbar = tqdm(range(len(train_loader)), desc=f"Epoch {epoch+1}/{config.epochs}")
        
        for i in pbar:
            batch = train_loader.one_batch()
            inputs, _ = batch
            x_ts, tabular = inputs
            x_cat, x_cont = tabular
            
            x_ts = x_ts.to(device)
            x_cat = x_cat.to(device) if x_cat is not None and x_cat.numel() > 0 else None
            x_cont = x_cont.to(device) if x_cont is not None and x_cont.numel() > 0 else None
            
            optimizer.zero_grad()
            
            losses = model(
                x_ts, x_cat, x_cont,
                return_contrastive=(config.contrastive_weight > 0)
            )
            loss = losses['total_loss']
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            # Update meters
            meters['total']['sum'] += loss.item()
            meters['total']['count'] += 1
            if 'ts_loss' in losses:
                meters['ts']['sum'] += losses['ts_loss'].item()
                meters['ts']['count'] += 1
            if 'cat_loss' in losses:
                meters['cat']['sum'] += losses['cat_loss'].item()
                meters['cat']['count'] += 1
            if 'cont_loss' in losses:
                meters['cont']['sum'] += losses['cont_loss'].item()
                meters['cont']['count'] += 1
            if 'contrastive_loss' in losses:
                meters['contrastive']['sum'] += losses['contrastive_loss'].item()
                meters['contrastive']['count'] += 1
            
            # Live postfix update
            postfix = {'Loss': f"{meters['total']['sum']/meters['total']['count']:.4f}",
                      'LR': f"{scheduler.get_last_lr()[0]:.2e}"}
            if meters['ts']['count'] > 0:
                postfix['TS'] = f"{meters['ts']['sum']/meters['ts']['count']:.4f}"
            if meters['cat']['count'] > 0:
                postfix['Cat'] = f"{meters['cat']['sum']/meters['cat']['count']:.4f}"
            if meters['cont']['count'] > 0:
                postfix['Cont'] = f"{meters['cont']['sum']/meters['cont']['count']:.4f}"
            if meters['contrastive']['count'] > 0:
                postfix['Cont'] = f"{meters['contrastive']['sum']/meters['contrastive']['count']:.4f}"
            
            pbar.set_postfix(postfix)
        
        # Average losses for history
        avg_train_loss = meters['total']['sum'] / meters['total']['count']
        history['train_loss'].append(avg_train_loss)
        history['ts_loss'].append(meters['ts']['sum'] / meters['ts']['count'] if meters['ts']['count'] > 0 else 0)
        history['cat_loss'].append(meters['cat']['sum'] / meters['cat']['count'] if meters['cat']['count'] > 0 else 0)
        history['cont_loss'].append(meters['cont']['sum'] / meters['cont']['count'] if meters['cont']['count'] > 0 else 0)
        history['contrastive_loss'].append(meters['contrastive']['sum'] / meters['contrastive']['count'] if meters['contrastive']['count'] > 0 else 0)
        history['lr'].append(scheduler.get_last_lr()[0])
        
        # Validation
        val_loss = None
        if val_loader is not None and (epoch + 1) % config.val_frequency == 0:
            model.eval()
            val_meters = {'total': {'sum': 0, 'count': 0}}
            val_pbar = tqdm(range(len(val_loader)), desc="Validating", leave=False)
            
            with torch.no_grad():
                for i in val_pbar:
                    batch = val_loader.one_batch()
                    inputs, _ = batch
                    x_ts, tabular = inputs
                    x_cat, x_cont = tabular
                    
                    x_ts = x_ts.to(device)
                    x_cat = x_cat.to(device) if x_cat is not None and x_cat.numel() > 0 else None
                    x_cont = x_cont.to(device) if x_cont is not None and x_cont.numel() > 0 else None
                    
                    losses = model(x_ts, x_cat, x_cont)
                    val_meters['total']['sum'] += losses['total_loss'].item()
                    val_meters['total']['count'] += 1
                    val_pbar.set_postfix({'ValLoss': f"{val_meters['total']['sum']/val_meters['total']['count']:.4f}"})
            
            val_loss = val_meters['total']['sum'] / val_meters['total']['count']
            history['val_loss'].append(val_loss)
            
            # Save best model
            if config.save_best and val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'config': config
                }, checkpoint_dir / 'best_model.pt')
                pbar.write(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")
            
            # Early stopping
            if early_stopping(val_loss):
                pbar.write(f"\nEarly stopping triggered at epoch {epoch+1}")
                break
        
        pbar.write(f"Epoch {epoch+1}/{config.epochs} - Train Loss: {avg_train_loss:.4f}")
        if val_loss is not None:
            pbar.write(f"  Val Loss: {val_loss:.4f}")
    
    # Plot training curves
    plot_training_curves(history)
    
    return history


def plot_training_curves(history):
    """Plot training curves."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Total loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    if history['val_loss']:
        axes[0, 0].plot(range(0, len(history['train_loss']), 
                              len(history['train_loss']) // len(history['val_loss'])),
                       history['val_loss'], label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Component losses
    if any(history['ts_loss']):
        axes[0, 1].plot(history['ts_loss'], label='TS Loss')
    if any(history['cat_loss']):
        axes[0, 1].plot(history['cat_loss'], label='Cat Loss')
    if any(history['cont_loss']):
        axes[0, 1].plot(history['cont_loss'], label='Cont Loss')
    if any(history['contrastive_loss']):
        axes[0, 1].plot(history['contrastive_loss'], label='Contrastive Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Component Losses')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning rate
    axes[1, 0].plot(history['lr'])
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Loss ratios
    if any(history['ts_loss']) and any(history['cat_loss']):
        ts_ratio = np.array(history['ts_loss']) / (np.array(history['cat_loss']) + 1e-8)
        axes[1, 1].plot(ts_ratio, label='TS/Cat Ratio')
    if any(history['cont_loss']) and any(history['cat_loss']):
        cont_ratio = np.array(history['cont_loss']) / (np.array(history['cat_loss']) + 1e-8)
        axes[1, 1].plot(cont_ratio, label='Cont/Cat Ratio')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss Ratio')
    axes[1, 1].set_title('Loss Component Ratios')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('models/figs/pretraining_curves.png', dpi=150, bbox_inches='tight')
    plt.show()


