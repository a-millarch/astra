
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List, Optional
import shap


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional

from astra.utils import logger, cfg
from astra.models.hybrid.training import get_backbone, Learner, patch_learner_get_preds
from astra.data.dataloader import prepare_data_and_dls

# ============================================================================
# Utility Functions 
# ============================================================================

def step_to_time(step):
    """Convert time step to actual time in minutes."""
    intervals = [
        {'start_h': 0, 'end_h': 6, 'bin_min': 10},
        {'start_h': 6, 'end_h': 12, 'bin_min': 20},
        {'start_h': 12, 'end_h': 24, 'bin_min': 60},
        {'start_h': 24, 'end_h': 72, 'bin_min': 240},
        {'start_h': 72, 'end_h': 336, 'bin_min': 720},
        {'start_h': 336, 'end_h': 720, 'bin_min': 1440},
        {'start_h': 720, 'end_h': 2160, 'bin_min': 10080},
        {'start_h': 2160, 'end_h': None, 'bin_min': 43200},
    ]
    bins_cum = [0]
    for interval in intervals[:-1]:
        duration_min = (interval['end_h'] - interval['start_h']) * 60
        bins = duration_min // interval['bin_min']
        bins_cum.append(bins_cum[-1] + bins)
    
    for i in range(len(bins_cum) - 1):
        if bins_cum[i] <= step < bins_cum[i+1]:
            interval = intervals[i]
            step_offset = step - bins_cum[i]
            start_min = interval['start_h'] * 60
            return start_min + (step_offset + 1) * interval['bin_min']
    return None


def time_to_hours(minutes):
    """Convert minutes to hours with proper formatting."""
    if minutes is None:
        return "N/A"
    hours = minutes / 60
    if hours < 24:
        return f"{hours:.1f}h"
    else:
        days = hours / 24
        return f"{days:.1f}d"


def create_channel_mapping(data):
    """Create channel to feature name mapping."""
    feature_values = (
        data["trainval"].complete
        .sort_values(['PID', 'FEATURE'])
        ['FEATURE']
        .drop_duplicates()
        .tolist()
    )
    
    channel2feature = {i: feat for i, feat in enumerate(feature_values)}
    feature2channel = {feat: i for i, feat in enumerate(feature_values)}
    
    return channel2feature, feature2channel


# ============================================================================
# Model Wrapper for SHAP (UPDATED for Multi-Hot Categorical TS)
# ============================================================================

class ModelWrapperWithEmbeddings(nn.Module):
    """
    Wrapper for SHAP compatibility with multi-hot categorical TS.
    
    Handles:
    - Continuous time series
    - Multi-hot categorical time series (NEW)
    - Static categorical (pre-embedded)
    - Static continuous
    """
    def __init__(self, model, has_cat_ts=False):
        super().__init__()
        self.model = model
        self.has_cat_ts = has_cat_ts
        
    def forward(self, x_ts, x_ts_cat_embedded=None, x_cat_embedded=None, x_cont=None):
        """
        Forward pass with pre-embedded features.
        
        Args:
            x_ts: Continuous time series [bs, c_in, seq_len]
            x_ts_cat_embedded: Pre-embedded categorical TS [bs, seq_len, d_model] (NEW)
            x_cat_embedded: Pre-embedded static categorical [bs, n_cat, d_model]
            x_cont: Static continuous [bs, n_cont]
        """
        device = x_ts.device
        
        # Handle NaN in time series
        mask = torch.isnan(x_ts)
        if mask.any():
            x_ts = x_ts.clone()
            x_ts[mask] = 0
        
        # Continuous time series encoding
        x = self.model.W_P(x_ts).transpose(1, 2)  # [bs, seq_len, d_model]
        
        # Add pre-embedded categorical TS (NEW)
        if self.has_cat_ts and x_ts_cat_embedded is not None:
            if self.model.cat_ts_combine == 'add':
                x = x + x_ts_cat_embedded
            else:  # 'concat'
                x = torch.cat([x, x_ts_cat_embedded], dim=-1)
        
        # Add pre-embedded static categorical
        if x_cat_embedded is not None and x_cat_embedded.shape[1] > 0:
            x = torch.cat([x, x_cat_embedded], 1)
        
        # Add static continuous
        if x_cont is not None and x_cont.shape[1] > 0:
            x_cont_emb = self.model.conv(x_cont.unsqueeze(1)).transpose(1, 2)
            x = torch.cat([x, x_cont_emb], 1)
        
        # Positional encoding
        x += self.model.pos_enc
        
        if self.model.res_drop is not None:
            x = self.model.res_drop(x)
        
        # Transformer
        x = self.model.transformer(x, key_padding_mask=None)
        
        # Head
        x = self.model.head(x)
        return x


def embed_categorical_ts(model, x_ts_cat, encoding_info):
    """
    Pre-embed multi-hot categorical time series (NEW).
    
    Args:
        model: TSTabFusionTransformerMultiHot
        x_ts_cat: [bs, n_categories, seq_len] - multi-hot encoded
        encoding_info: From MultiHotCategoricalEncoder
    
    Returns:
        embedded: [bs, seq_len, d_model] - embedded categorical TS
    """
    if x_ts_cat is None or model.n_ts_cat == 0:
        return None
    
    # Convert TSTensor if needed
    if hasattr(x_ts_cat, 'data'):
        x_ts_cat = x_ts_cat.data
    x_ts_cat = x_ts_cat.float()
    
    # Transpose to [bs, seq_len, n_categories]
    x_ts_cat = x_ts_cat.transpose(1, 2)
    
    with torch.no_grad():
        x_ts_cat_embedded_list = []
        dim_offset = 0
        
        for embed_layer, (feat_name, n_classes) in zip(
            model.ts_cat_embeds, model.ts_cat_dims.items()
        ):
            feat_multi_hot = x_ts_cat[:, :, dim_offset:dim_offset + n_classes]
            feat_embedded = embed_layer(feat_multi_hot)  # [bs, seq_len, d_model]
            x_ts_cat_embedded_list.append(feat_embedded)
            dim_offset += n_classes
        
        # Sum embeddings (as in forward pass)
        if model.cat_ts_combine == 'add':
            x_ts_cat_embedded = torch.stack(x_ts_cat_embedded_list, dim=0).sum(dim=0)
        else:  # 'concat'
            x_ts_cat_embedded = torch.cat(x_ts_cat_embedded_list, dim=-1)
    
    x_ts_cat_embedded.requires_grad = True
    return x_ts_cat_embedded


def embed_categorical_features(model, x_cat):
    """Pre-embed static categorical features."""
    if x_cat is None or x_cat.shape[1] == 0:
        return None
    
    with torch.no_grad():
        x_cat_emb = [model.embeds[i](x_cat[:, i]).unsqueeze(1) for i in range(x_cat.shape[1])]
        x_cat_emb = torch.cat(x_cat_emb, 1)
    
    x_cat_emb.requires_grad = True
    return x_cat_emb


# ============================================================================
# Data Extraction (UPDATED for TSAI format with categorical TS)
# ============================================================================

def extract_data_from_dataloader(dataloader, max_samples=None, device='cpu'):
    """
    Extract data from TSAI MixedDataLoader with categorical TS support.
    
    Returns:
        Tuple of (x_ts, x_ts_cat, x_cat, x_cont, y)
    """
    all_ts = []
    all_ts_cat = []
    all_cat = []
    all_cont = []
    all_y = []
    
    n_samples = 0
    
    for batch in dataloader:
        if max_samples is not None and n_samples >= max_samples:
            break
        
        # Unpack TSAI format: ((x_ts, x_tab, x_ts_cat), y)
        inputs, targets = batch
        
        x_ts = inputs[0]           # Continuous TS
        x_tab = inputs[1]          # Tabular (tuple)
        x_ts_cat = inputs[2]       # Categorical TS (NEW)
        
        x_cat, x_cont = x_tab
        
        all_ts.append(x_ts.cpu())
        all_ts_cat.append(x_ts_cat.cpu())
        all_cat.append(x_cat.cpu())
        all_cont.append(x_cont.cpu())
        all_y.append(targets.cpu())
        
        n_samples += x_ts.shape[0]
    
    x_ts_full = torch.cat(all_ts, dim=0)
    x_ts_cat_full = torch.cat(all_ts_cat, dim=0)
    x_cat_full = torch.cat(all_cat, dim=0)
    x_cont_full = torch.cat(all_cont, dim=0)
    y_full = torch.cat(all_y, dim=0)
    
    if max_samples is not None and x_ts_full.shape[0] > max_samples:
        x_ts_full = x_ts_full[:max_samples]
        x_ts_cat_full = x_ts_cat_full[:max_samples]
        x_cat_full = x_cat_full[:max_samples]
        x_cont_full = x_cont_full[:max_samples]
        y_full = y_full[:max_samples]
    
    x_ts_full = x_ts_full.to(device)
    x_ts_cat_full = x_ts_cat_full.to(device)
    x_cat_full = x_cat_full.to(device)
    x_cont_full = x_cont_full.to(device)
    y_full = y_full.to(device)
    
    return x_ts_full, x_ts_cat_full, x_cat_full, x_cont_full, y_full


# ============================================================================
# SHAP Calculation (UPDATED)
# ============================================================================

def calculate_shap_from_dataloaders(
    model,
    background_loader,
    test_loader,
    encoding_info,  # NEW: Required for categorical TS
    device='cuda',
    max_background_samples=200,
    max_test_samples=100
):
    """
    Calculate SHAP values with multi-hot categorical TS support.
    
    Args:
        model: TSTabFusionTransformerMultiHot
        background_loader: Training data loader
        test_loader: Test data loader
        encoding_info: From MultiHotCategoricalEncoder (NEW)
        device: Device to use
        max_background_samples: Number of background samples
        max_test_samples: Number of test samples
    
    Returns:
        Dictionary with SHAP values for all modalities
    """
    print("Extracting background data...")
    bg_ts, bg_ts_cat, bg_cat, bg_cont, bg_y = extract_data_from_dataloader(
        background_loader, 
        max_samples=max_background_samples,
        device=device
    )
    print(f"  Background: {bg_ts.shape[0]} samples")
    print(f"    Continuous TS: {bg_ts.shape}")
    print(f"    Categorical TS: {bg_ts_cat.shape}")
    
    print("Extracting test data...")
    test_ts, test_ts_cat, test_cat, test_cont, test_y = extract_data_from_dataloader(
        test_loader,
        max_samples=max_test_samples,
        device=device
    )
    print(f"  Test: {test_ts.shape[0]} samples")
    
    model.eval()
    model = model.to(device)
    
    # Pre-embed features
    print("Pre-embedding categorical features...")
    
    # Categorical TS (NEW)
    has_cat_ts = model.n_ts_cat > 0 and bg_ts_cat is not None and bg_ts_cat.numel() > 0
    if has_cat_ts:
        bg_ts_cat_emb = embed_categorical_ts(model, bg_ts_cat, encoding_info)
        test_ts_cat_emb = embed_categorical_ts(model, test_ts_cat, encoding_info)
        print(f"  Categorical TS embedded: {bg_ts_cat_emb.shape}")
    else:
        bg_ts_cat_emb = None
        test_ts_cat_emb = None
    
    # Static categorical
    if bg_cat.shape[1] > 0:
        bg_cat_emb = embed_categorical_features(model, bg_cat)
        test_cat_emb = embed_categorical_features(model, test_cat)
        print(f"  Static categorical embedded: {bg_cat_emb.shape}")
    else:
        bg_cat_emb = None
        test_cat_emb = None
    
    # Create wrapper
    wrapped_model = ModelWrapperWithEmbeddings(model, has_cat_ts=has_cat_ts)
    
    print("Creating SHAP GradientExplainer...")
    
    # Build input lists
    bg_inputs = [bg_ts]
    test_inputs = [test_ts]
    
    if has_cat_ts:
        bg_inputs.append(bg_ts_cat_emb)
        test_inputs.append(test_ts_cat_emb)
    
    if bg_cat_emb is not None:
        bg_inputs.append(bg_cat_emb)
        test_inputs.append(test_cat_emb)
    
    if bg_cont.shape[1] > 0:
        bg_inputs.append(bg_cont)
        test_inputs.append(test_cont)
    
    explainer = shap.GradientExplainer(wrapped_model, bg_inputs)
    
    print("Calculating SHAP values (this may take a while)...")
    shap_values = explainer.shap_values(test_inputs)
    
    print("SHAP calculation complete!")
    
    # Unpack SHAP values
    if isinstance(shap_values, list) and len(shap_values) > 0:
        if isinstance(shap_values[0], list):
            shap_values = shap_values[0]
    
    idx = 0
    ts_shap = shap_values[idx]
    idx += 1
    
    # Categorical TS SHAP (NEW)
    if has_cat_ts:
        cat_ts_shap_embedded = shap_values[idx]
        # Average over embedding dimension to get per-timestep importance
        cat_ts_shap = np.abs(cat_ts_shap_embedded).mean(axis=2)  # [n_samples, seq_len]
        idx += 1
    else:
        cat_ts_shap = None
        cat_ts_shap_embedded = None
    
    # Static categorical SHAP
    if bg_cat_emb is not None:
        cat_shap_embedded = shap_values[idx]
        cat_shap = np.abs(cat_shap_embedded).mean(axis=2)
        idx += 1
    else:
        cat_shap = None
        cat_shap_embedded = None
    
    # Static continuous SHAP
    if bg_cont.shape[1] > 0:
        cont_shap = shap_values[idx]
    else:
        cont_shap = None
    
    results = {
        'ts_shap': ts_shap,
        'cat_ts_shap': cat_ts_shap,  # NEW
        'cat_ts_shap_embedded': cat_ts_shap_embedded,  # NEW
        'cat_shap': cat_shap,
        'cat_shap_embedded': cat_shap_embedded,
        'cont_shap': cont_shap,
        'test_data': {
            'ts': test_ts.cpu().numpy(),
            'ts_cat': test_ts_cat.cpu().numpy(),  # NEW
            'cat': test_cat.cpu().numpy(),
            'cont': test_cont.cpu().numpy(),
            'y': test_y.cpu().numpy()
        },
        'background_data': {
            'ts': bg_ts.cpu().numpy(),
            'ts_cat': bg_ts_cat.cpu().numpy(),  # NEW
            'cat': bg_cat.cpu().numpy(),
            'cont': bg_cont.cpu().numpy(),
            'y': bg_y.cpu().numpy()
        },
        'encoding_info': encoding_info  # NEW
    }
    
    return results



def visualize_shap_individual(
    shap_results: Dict,
    sample_idx: int = 0,
    channel2feature: Dict[int, str] = None,
    feature_names_cat: List[str] = None,
    feature_names_cont: List[str] = None,
    class_idx: int = 1,
    save_path: str = None
):
    """
    Visualize SHAP values for individual with multi-hot categorical TS.
    
    NEW: Shows medication/diagnosis importance over time
    """
    fig = plt.figure(figsize=(22, 18))
    gs = fig.add_gridspec(5, 2, hspace=0.4, wspace=0.3, height_ratios=[1, 1, 0.8, 1, 1])
    
    ts_shap = shap_results['ts_shap'][sample_idx]
    ts_data = shap_results['test_data']['ts'][sample_idx]
    
    # Handle multi-class
    if ts_shap.ndim == 3:
        ts_shap = ts_shap[:, :, class_idx]
    
    n_channels, n_steps = ts_shap.shape
    
    # Time labels
    time_labels = [step_to_time(i) for i in range(n_steps)]
    time_labels_formatted = [time_to_hours(t) for t in time_labels]
    n_ticks = min(10, n_steps)
    tick_indices = np.linspace(0, n_steps-1, n_ticks, dtype=int)
    
    # === Plot 1: Continuous TS SHAP over time ===
    ax1 = fig.add_subplot(gs[0, :])
    ts_shap_avg = np.abs(ts_shap).mean(axis=0)
    
    ax1.plot(ts_shap_avg, linewidth=2, color='#ff0051', label='Continuous TS')
    ax1.fill_between(range(len(ts_shap_avg)), ts_shap_avg, alpha=0.3, color='#ff0051')
    
    # Add categorical TS if available (NEW)
    if shap_results['cat_ts_shap'] is not None:
        cat_ts_shap = shap_results['cat_ts_shap'][sample_idx]
        if cat_ts_shap.ndim == 2:
            cat_ts_shap = cat_ts_shap[:, class_idx]
        ax1.plot(cat_ts_shap, linewidth=2, color='#00d4aa', label='Categorical TS', linestyle='--')
        ax1.fill_between(range(len(cat_ts_shap)), cat_ts_shap, alpha=0.2, color='#00d4aa')
    
    ax1.set_xlabel('Time', fontsize=12)
    ax1.set_ylabel('|SHAP Value|', fontsize=12)
    ax1.set_title(f'Time Series SHAP Importance Over Time (Sample {sample_idx}, Class {class_idx})', 
                  fontsize=14, fontweight='bold')
    ax1.set_xticks(tick_indices)
    ax1.set_xticklabels([time_labels_formatted[i] for i in tick_indices], rotation=45)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # === Plot 2: Continuous TS heatmap ===
    ax2 = fig.add_subplot(gs[1, :])
    im = ax2.imshow(ts_shap, aspect='auto', cmap='RdBu_r', interpolation='nearest')
    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_ylabel('Channel', fontsize=12)
    ax2.set_title('Continuous Time Series SHAP Heatmap', fontsize=14, fontweight='bold')
    
    if channel2feature is not None:
        channel_labels = [channel2feature.get(i, f'Ch{i}') for i in range(n_channels)]
        if n_channels > 20:
            tick_step = n_channels // 20
            y_ticks = list(range(0, n_channels, tick_step))
            y_labels = [channel_labels[i] for i in y_ticks]
            ax2.set_yticks(y_ticks)
            ax2.set_yticklabels(y_labels, fontsize=8)
        else:
            ax2.set_yticks(range(n_channels))
            ax2.set_yticklabels(channel_labels, fontsize=9)
    
    ax2.set_xticks(tick_indices)
    ax2.set_xticklabels([time_labels_formatted[i] for i in tick_indices], rotation=45)
    plt.colorbar(im, ax=ax2, label='SHAP Value')
    
    # === Plot 3: NEW - Categorical TS (Medications/Diagnoses) over time ===
    if shap_results['cat_ts_shap'] is not None and 'encoding_info' in shap_results:
        ax3 = fig.add_subplot(gs[2, :])
        
        # Get original multi-hot data
        cat_ts_data = shap_results['test_data']['ts_cat'][sample_idx]  # [n_categories, seq_len]
        encoding_info = shap_results['encoding_info']
        
        # For each feature (e.g., medication), show which ones are active over time
        dim_offset = 0
        colors = plt.cm.Set3(np.linspace(0, 1, len(encoding_info['feature_ranges'])))
        
        for feat_idx, (feat_name, (start, end)) in enumerate(encoding_info['feature_ranges'].items()):
            n_classes = end - start
            feat_data = cat_ts_data[start:end, :]  # [n_classes, seq_len]
            
            # Find timesteps where any category is active
            active_timesteps = np.where(feat_data.sum(axis=0) > 0)[0]
            
            if len(active_timesteps) > 0:
                # Plot as scatter
                y_pos = np.ones(len(active_timesteps)) * feat_idx
                ax3.scatter(active_timesteps, y_pos, alpha=0.6, s=50, 
                           color=colors[feat_idx], label=feat_name, marker='|')
        
        ax3.set_xlabel('Time', fontsize=12)
        ax3.set_ylabel('Feature', fontsize=12)
        ax3.set_title('Categorical TS Activity Timeline (Medications/Diagnoses)', 
                     fontsize=14, fontweight='bold')
        ax3.set_xticks(tick_indices)
        ax3.set_xticklabels([time_labels_formatted[i] for i in tick_indices], rotation=45)
        ax3.set_yticks(range(len(encoding_info['feature_ranges'])))
        ax3.set_yticklabels(list(encoding_info['feature_ranges'].keys()))
        ax3.legend(loc='upper right', fontsize=9)
        ax3.grid(True, alpha=0.3, axis='x')
    
    # === Plot 4: Channel importance ===
    ax4 = fig.add_subplot(gs[3, :])
    channel_importance = np.abs(ts_shap).mean(axis=1)
    sorted_idx = np.argsort(channel_importance)[::-1]
    sorted_importance = channel_importance[sorted_idx]
    
    if channel2feature is not None:
        sorted_names = [channel2feature.get(i, f'Ch{i}') for i in sorted_idx]
    else:
        sorted_names = [f'Channel {i}' for i in sorted_idx]
    
    n_show = min(20, len(sorted_importance))
    ax4.barh(range(n_show), sorted_importance[:n_show], color='#008bfb', alpha=0.7)
    ax4.set_yticks(range(n_show))
    ax4.set_yticklabels(sorted_names[:n_show], fontsize=9)
    ax4.set_xlabel('Mean |SHAP Value| Across Time', fontsize=12)
    ax4.set_title(f'Top {n_show} Most Important Continuous Channels', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x')
    ax4.invert_yaxis()
    
    # === Plot 5 & 6: Static categorical and continuous ===
    if shap_results['cat_shap'] is not None and shap_results['cat_shap'].size > 0:
        ax5 = fig.add_subplot(gs[4, 0])
        cat_shap = shap_results['cat_shap'][sample_idx]
        cat_data = shap_results['test_data']['cat'][sample_idx]
        
        if cat_shap.ndim == 2:
            cat_shap = cat_shap[:, class_idx]
        
        if feature_names_cat is None:
            feature_names_cat = [f'Cat_{i}' for i in range(len(cat_shap))]
        
        n_cat = len(cat_shap)
        feature_names_cat = (list(feature_names_cat) + [f'Cat_{i}' for i in range(len(feature_names_cat), n_cat)])[:n_cat]
        
        colors = ['#ff0051' if x > 0 else '#008bfb' for x in cat_shap]
        ax5.barh(range(len(cat_shap)), cat_shap, color=colors, alpha=0.7)
        ax5.set_yticks(range(len(cat_shap)))
        ax5.set_yticklabels([f'{name}\n(val={int(val)})' 
                            for name, val in zip(feature_names_cat, cat_data)], fontsize=9)
        ax5.set_xlabel('SHAP Value', fontsize=12)
        ax5.set_title('Static Categorical Features', fontsize=14, fontweight='bold')
        ax5.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax5.grid(True, alpha=0.3, axis='x')
        ax5.invert_yaxis()
    
    if shap_results['cont_shap'] is not None and shap_results['cont_shap'].size > 0:
        ax6 = fig.add_subplot(gs[4, 1])
        cont_shap = shap_results['cont_shap'][sample_idx]
        cont_data = shap_results['test_data']['cont'][sample_idx]
        
        if cont_shap.ndim == 2:
            cont_shap = cont_shap[:, class_idx]
        
        if feature_names_cont is None:
            feature_names_cont = [f'Cont_{i}' for i in range(len(cont_shap))]
        
        n_cont = len(cont_shap)
        feature_names_cont = (list(feature_names_cont) + [f'Cont_{i}' for i in range(len(feature_names_cont), n_cont)])[:n_cont]
        
        colors = ['#ff0051' if x > 0 else '#008bfb' for x in cont_shap]
        ax6.barh(range(len(cont_shap)), cont_shap, color=colors, alpha=0.7)
        ax6.set_yticks(range(len(cont_shap)))
        ax6.set_yticklabels([f'{name}\n(val={val:.2f})' 
                            for name, val in zip(feature_names_cont, cont_data)], fontsize=9)
        ax6.set_xlabel('SHAP Value', fontsize=12)
        ax6.set_title('Static Continuous Features', fontsize=14, fontweight='bold')
        ax6.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax6.grid(True, alpha=0.3, axis='x')
        ax6.invert_yaxis()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def visualize_shap_summary(
    shap_results: Dict,
    channel2feature: Dict[int, str] = None,
    feature_names_cat: List[str] = None,
    feature_names_cont: List[str] = None,
    max_display: int = 20,
    class_idx: int = 1,
    save_path: str = None
):
    """
    Summary visualizations with multi-hot categorical TS.
    
    NEW: Shows medication/diagnosis importance patterns across cohort
    """
    fig = plt.figure(figsize=(22, 16))
    gs = fig.add_gridspec(4, 2, hspace=0.4, wspace=0.3, height_ratios=[1, 1, 1, 1])
    
    ts_shap = shap_results['ts_shap']
    
    if ts_shap.ndim == 4:
        ts_shap = ts_shap[:, :, :, class_idx]
    
    n_samples, n_channels, n_steps = ts_shap.shape
    
    # Time labels
    time_labels = [step_to_time(i) for i in range(n_steps)]
    time_labels_formatted = [time_to_hours(t) for t in time_labels]
    n_ticks = min(10, n_steps)
    tick_indices = np.linspace(0, n_steps-1, n_ticks, dtype=int)
    
    # === Plot 1: Continuous vs Categorical TS importance over time (NEW) ===
    ax1 = fig.add_subplot(gs[0, :])
    ts_importance = np.abs(ts_shap).mean(axis=(0, 1))
    
    ax1.plot(ts_importance, linewidth=2, color='#ff0051', label='Continuous TS')
    ax1.fill_between(range(len(ts_importance)), ts_importance, alpha=0.3, color='#ff0051')
    
    # Add categorical TS if available (NEW)
    if shap_results['cat_ts_shap'] is not None:
        cat_ts_shap = shap_results['cat_ts_shap']
        if cat_ts_shap.ndim == 3:
            cat_ts_shap = cat_ts_shap[:, :, class_idx]
        cat_ts_importance = np.abs(cat_ts_shap).mean(axis=0)
        
        ax1.plot(cat_ts_importance, linewidth=2, color='#00d4aa', 
                label='Categorical TS (Medications/Diagnoses)', linestyle='--')
        ax1.fill_between(range(len(cat_ts_importance)), cat_ts_importance, 
                        alpha=0.2, color='#00d4aa')
    
    ax1.set_xlabel('Time', fontsize=12)
    ax1.set_ylabel('Mean |SHAP Value|', fontsize=12)
    ax1.set_title(f'Feature Importance Over Time - Continuous vs Categorical (Class {class_idx})', 
                  fontsize=14, fontweight='bold')
    ax1.set_xticks(tick_indices)
    ax1.set_xticklabels([time_labels_formatted[i] for i in tick_indices], rotation=45)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # === Plot 2: Channel importance ===
    ax2 = fig.add_subplot(gs[1, 0])
    channel_importance = np.abs(ts_shap).mean(axis=(0, 2))
    sorted_idx = np.argsort(channel_importance)[::-1][:max_display]
    sorted_importance = channel_importance[sorted_idx]
    
    if channel2feature is not None:
        sorted_names = [channel2feature.get(int(i), f'Ch{i}') for i in sorted_idx]
    else:
        sorted_names = [f'Channel {i}' for i in sorted_idx]
    
    ax2.barh(range(len(sorted_importance)), sorted_importance, color='#008bfb', alpha=0.7)
    ax2.set_yticks(range(len(sorted_importance)))
    ax2.set_yticklabels(sorted_names, fontsize=10)
    ax2.set_xlabel('Mean |SHAP Value|', fontsize=12)
    ax2.set_title(f'Top {len(sorted_importance)} Continuous Channels', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.invert_yaxis()
    
    # === Plot 3: NEW - Categorical TS feature importance ===
    if shap_results['cat_ts_shap'] is not None and 'encoding_info' in shap_results:
        ax3 = fig.add_subplot(gs[1, 1])
        
        cat_ts_shap = shap_results['cat_ts_shap']
        if cat_ts_shap.ndim == 3:
            cat_ts_shap = cat_ts_shap[:, :, class_idx]
        
        # Average importance across samples and time
        cat_ts_importance = np.abs(cat_ts_shap).mean(axis=(0, 1))
        
        encoding_info = shap_results['encoding_info']
        feat_names = list(encoding_info['feature_ranges'].keys())
        
        ax3.barh(range(len(feat_names)), [cat_ts_importance] * len(feat_names), 
                color='#00d4aa', alpha=0.7)
        ax3.set_yticks(range(len(feat_names)))
        ax3.set_yticklabels(feat_names, fontsize=10)
        ax3.set_xlabel('Mean |SHAP Value|', fontsize=12)
        ax3.set_title('Categorical TS Features (Medications/Diagnoses)', 
                     fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='x')
        ax3.invert_yaxis()
    
    # === Plot 4: Continuous TS heatmap ===
    ax4 = fig.add_subplot(gs[2, :])
    ts_shap_mean = np.abs(ts_shap).mean(axis=0)
    
    im = ax4.imshow(ts_shap_mean, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    ax4.set_xlabel('Time', fontsize=12)
    ax4.set_ylabel('Channel', fontsize=12)
    ax4.set_title('Continuous TS SHAP Heatmap (Mean Across Cohort)', fontsize=14, fontweight='bold')
    
    if channel2feature is not None:
        channel_labels = [channel2feature.get(i, f'Ch{i}') for i in range(n_channels)]
        if n_channels > 25:
            tick_step = max(1, n_channels // 25)
            y_ticks = list(range(0, n_channels, tick_step))
            y_labels = [channel_labels[i] for i in y_ticks]
            ax4.set_yticks(y_ticks)
            ax4.set_yticklabels(y_labels, fontsize=8)
        else:
            ax4.set_yticks(range(n_channels))
            ax4.set_yticklabels(channel_labels, fontsize=9)
    
    ax4.set_xticks(tick_indices)
    ax4.set_xticklabels([time_labels_formatted[i] for i in tick_indices], rotation=45)
    plt.colorbar(im, ax=ax4, label='Mean |SHAP Value|')
    
    # === Plot 5 & 6: Static features ===
    if shap_results['cat_shap'] is not None and shap_results['cat_shap'].size > 0:
        ax5 = fig.add_subplot(gs[3, 0])
        cat_shap = shap_results['cat_shap']
        
        if cat_shap.ndim == 3:
            cat_shap = cat_shap[:, :, class_idx]
        
        cat_importance = np.abs(cat_shap).mean(axis=0) if cat_shap.ndim > 1 else np.abs(cat_shap)
        
        if feature_names_cat is None:
            feature_names_cat = [f'Cat_{i}' for i in range(len(cat_importance))]
        
        n_features = min(len(cat_importance), len(feature_names_cat))
        sorted_idx = np.argsort(cat_importance[:n_features])[::-1][:max_display]
        sorted_importance = cat_importance[sorted_idx]
        sorted_names = [feature_names_cat[int(i)] for i in sorted_idx]
        
        ax5.barh(range(len(sorted_importance)), sorted_importance, color='#ff0051', alpha=0.7)
        ax5.set_yticks(range(len(sorted_importance)))
        ax5.set_yticklabels(sorted_names, fontsize=10)
        ax5.set_xlabel('Mean |SHAP Value|', fontsize=12)
        ax5.set_title(f'Top {len(sorted_importance)} Static Categorical', fontsize=14, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='x')
        ax5.invert_yaxis()
    
    if shap_results['cont_shap'] is not None and shap_results['cont_shap'].size > 0:
        ax6 = fig.add_subplot(gs[3, 1])
        cont_shap = shap_results['cont_shap']
        
        if cont_shap.ndim == 3:
            cont_shap = cont_shap[:, :, class_idx]
        
        cont_importance = np.abs(cont_shap).mean(axis=0) if cont_shap.ndim > 1 else np.abs(cont_shap)
        
        if feature_names_cont is None:
            feature_names_cont = [f'Cont_{i}' for i in range(len(cont_importance))]
        
        n_features = min(len(cont_importance), len(feature_names_cont))
        sorted_idx = np.argsort(cont_importance[:n_features])[::-1][:max_display]
        sorted_importance = cont_importance[sorted_idx]
        sorted_names = [feature_names_cont[int(i)] for i in sorted_idx]
        
        ax6.barh(range(len(sorted_importance)), sorted_importance, color='#008bfb', alpha=0.7)
        ax6.set_yticks(range(len(sorted_importance)))
        ax6.set_yticklabels(sorted_names, fontsize=10)
        ax6.set_xlabel('Mean |SHAP Value|', fontsize=12)
        ax6.set_title(f'Top {len(sorted_importance)} Static Continuous', fontsize=14, fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='x')
        ax6.invert_yaxis()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def create_channel_mapping(data):
    """
    Create channel to feature name mapping.
    
    Args:
        data: Dictionary with 'trainval' key containing data with FEATURE column
        
    Returns:
        Tuple of (channel2feature dict, feature2channel dict)
    """
    feature_values = (
        data["trainval"].complete
        .sort_values(['PID', 'FEATURE'])
        ['FEATURE']
        .drop_duplicates()
        .tolist()
    )
    
    channel2feature = {i: feat for i, feat in enumerate(feature_values)}
    feature2channel = {feat: i for i, feat in enumerate(feature_values)}
    
    return channel2feature, feature2channel


def prepare_learner(data, model_name='13012025'):

    mixed_dls = data["mixed_dls"]
    holdout = data["holdout"]
    
    # ============================================================================
    # LOAD MODEL
    # ============================================================================
    logger.info(f"Loading model: {model_name}")
    backbone = get_backbone(data, cfg)
    learn = Learner(mixed_dls, backbone, metrics=None)
    
    # DON'T assign return value - it returns the model, not the learner!
    learn.load(model_name)
    learn.to('cuda')
    learn = patch_learner_get_preds(learn)
    logger.info("Model loaded and moved to GPU")
    return learn


# ============================================================================
def shap_analysis(data=None, learn=None, model_name='13012025'):
    """
    get data, calculate shap, visualize both cohort and example individual for retrospective data
    TODO: get model cfg from cfg file
    """
    if data is None:
        data = prepare_data_and_dls()
        
    if learn is None:
        learn= prepare_learner(data, model_name)

    shap_results = calculate_shap_from_dataloaders(
                                model=learn.model,
                                background_loader=data["mixed_dls"].train,
                                test_loader=data["holdout_mixed_dls"].train,
                                encoding_info = data["encoding_info"],
                                device='cuda',
                                max_background_samples=6000,
                                max_test_samples=900
                            )
    channel2feature, feature2channel = create_channel_mapping(data)

    # Visualize cohort summary for positive class
    visualize_shap_summary(
        shap_results,
        channel2feature=channel2feature,        
        feature_names_cat=cfg["dataset"]["cat_cols"],
        feature_names_cont=cfg["dataset"]["num_cols"],
        class_idx=1,
        max_display=20,
        save_path=f'reports/shap_summary_cohort_{model_name}.png'
    )

    # Visualize for a specific observation
    visualize_shap_individual(
        shap_results,
        sample_idx=1,
        channel2feature=channel2feature,
        feature_names_cat=cfg["dataset"]["cat_cols"],
        feature_names_cont=cfg["dataset"]["num_cols"],
        class_idx=1,
        save_path=f'reports/shap_individual_sample_1_{model_name}.png'
    )
    return shap_results, channel2feature, feature2channel 
