from astra.models.hybrid.train_model import *
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List, Optional
import shap

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


# ============================================================================
# Model Wrapper for SHAP
# ============================================================================

class ModelWrapperWithEmbeddings(nn.Module):
    """Wrapper that extracts embeddings first for SHAP compatibility."""
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x_ts, x_cat_embedded, x_cont):
        """Forward pass with pre-embedded categorical features."""
        device = x_ts.device
        
        # Handle NaN in time series
        mask = torch.isnan(x_ts)
        if mask.any():
            x_ts = x_ts.clone()
            x_ts[mask] = 0
        
        # Time series encoding
        x = self.model.W_P(x_ts).transpose(1, 2)
        
        # Concatenate pre-embedded categorical features
        if x_cat_embedded is not None and x_cat_embedded.shape[1] > 0:
            x = torch.cat([x, x_cat_embedded], 1)
        
        # Continuous encoding
        if x_cont is not None and x_cont.shape[1] > 0:
            x_cont_emb = self.model.conv(x_cont.unsqueeze(1)).transpose(1, 2)
            x = torch.cat([x, x_cont_emb], 1)
        
        # Positional encoding
        x += self.model.pos_enc
        
        if self.model.res_drop is not None:
            x = self.model.res_drop(x)
        
        # Transformer
        import inspect
        transformer_sig = inspect.signature(self.model.transformer.forward)
        if 'key_padding_mask' in transformer_sig.parameters:
            x = self.model.transformer(x, None, None)
        else:
            x = self.model.transformer(x, None)
        
        # Head
        x = self.model.head(x)
        return x


def embed_categorical_features(model, x_cat):
    """Pre-embed categorical features to avoid gradient issues with SHAP."""
    if x_cat is None or x_cat.shape[1] == 0:
        return None
    
    with torch.no_grad():
        x_cat_emb = [model.embeds[i](x_cat[:, i]).unsqueeze(1) for i in range(x_cat.shape[1])]
        x_cat_emb = torch.cat(x_cat_emb, 1)
    
    x_cat_emb.requires_grad = True
    return x_cat_emb


# ============================================================================
# Data Extraction
# ============================================================================

def extract_data_from_dataloader(dataloader, max_samples=None, device='cpu'):
    """Extract data from TSAI MixedDataLoader."""
    all_ts = []
    all_cat = []
    all_cont = []
    all_y = []
    
    n_samples = 0
    
    for i in range(len(dataloader)):
        if max_samples is not None and n_samples >= max_samples:
            break
            
        batch = dataloader.one_batch()
        inputs, targets = batch
        x_ts, tabular = inputs
        x_cat, x_cont = tabular
        
        all_ts.append(x_ts.cpu())
        all_cat.append(x_cat.cpu())
        all_cont.append(x_cont.cpu())
        all_y.append(targets.cpu())
        
        n_samples += x_ts.shape[0]
    
    x_ts_full = torch.cat(all_ts, dim=0)
    x_cat_full = torch.cat(all_cat, dim=0)
    x_cont_full = torch.cat(all_cont, dim=0)
    y_full = torch.cat(all_y, dim=0)
    
    if max_samples is not None and x_ts_full.shape[0] > max_samples:
        x_ts_full = x_ts_full[:max_samples]
        x_cat_full = x_cat_full[:max_samples]
        x_cont_full = x_cont_full[:max_samples]
        y_full = y_full[:max_samples]
    
    x_ts_full = x_ts_full.to(device)
    x_cat_full = x_cat_full.to(device)
    x_cont_full = x_cont_full.to(device)
    y_full = y_full.to(device)
    
    return x_ts_full, x_cat_full, x_cont_full, y_full


# ============================================================================
# SHAP Calculation
# ============================================================================

def calculate_shap_from_dataloaders(
    model,
    background_loader,
    test_loader,
    device='cuda',
    max_background_samples=200,
    max_test_samples=100
):
    """Calculate SHAP values using TSAI MixedDataLoaders."""
    print("Extracting background data from trainval loader...")
    bg_ts, bg_cat, bg_cont, bg_y = extract_data_from_dataloader(
        background_loader, 
        max_samples=max_background_samples,
        device=device
    )
    print(f"  Background data: {bg_ts.shape[0]} samples")
    
    print("Extracting test data from holdout loader...")
    test_ts, test_cat, test_cont, test_y = extract_data_from_dataloader(
        test_loader,
        max_samples=max_test_samples,
        device=device
    )
    print(f"  Test data: {test_ts.shape[0]} samples")
    
    model.eval()
    model = model.to(device)
    
    print("Pre-embedding categorical features...")
    bg_cat_emb = embed_categorical_features(model, bg_cat) if bg_cat.shape[1] > 0 else None
    test_cat_emb = embed_categorical_features(model, test_cat) if test_cat.shape[1] > 0 else None
    
    wrapped_model = ModelWrapperWithEmbeddings(model)
    
    print("Creating SHAP GradientExplainer...")
    
    if bg_cat_emb is not None:
        bg_inputs = [bg_ts, bg_cat_emb, bg_cont]
        test_inputs = [test_ts, test_cat_emb, test_cont]
    else:
        bg_inputs = [bg_ts, bg_cont]
        test_inputs = [test_ts, test_cont]
    
    explainer = shap.GradientExplainer(wrapped_model, bg_inputs)
    
    print("Calculating SHAP values (this may take a while)...")
    shap_values = explainer.shap_values(test_inputs)
    
    print("SHAP calculation complete!")
    
    if isinstance(shap_values, list) and len(shap_values) > 0:
        if isinstance(shap_values[0], list):
            shap_values = shap_values[0]
    
    ts_shap = shap_values[0]
    
    if bg_cat_emb is not None:
        cat_shap_embedded = shap_values[1]
        cat_shap = np.abs(cat_shap_embedded).mean(axis=2)
        cont_shap = shap_values[2] if len(shap_values) > 2 else None
    else:
        cat_shap = None
        cont_shap = shap_values[1] if len(shap_values) > 1 else None
    
    results = {
        'ts_shap': ts_shap,
        'cat_shap': cat_shap,
        'cont_shap': cont_shap,
        'cat_shap_embedded': cat_shap_embedded if bg_cat_emb is not None else None,
        'test_data': {
            'ts': test_ts.cpu().numpy(),
            'cat': test_cat.cpu().numpy(),
            'cont': test_cont.cpu().numpy(),
            'y': test_y.cpu().numpy()
        },
        'background_data': {
            'ts': bg_ts.cpu().numpy(),
            'cat': bg_cat.cpu().numpy(),
            'cont': bg_cont.cpu().numpy(),
            'y': bg_y.cpu().numpy()
        }
    }
    
    return results


# ============================================================================
# Visualization Functions
# ============================================================================

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
    Visualize SHAP values for a single observation with enhanced channel mapping.
    """
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.3, height_ratios=[1, 1, 1, 1])
    
    ts_shap = shap_results['ts_shap'][sample_idx]
    ts_data = shap_results['test_data']['ts'][sample_idx]
    
    # Handle multi-class output
    if ts_shap.ndim == 3:
        ts_shap = ts_shap[:, :, class_idx]
    
    n_channels, n_steps = ts_shap.shape
    
    # Create time labels
    time_labels = [step_to_time(i) for i in range(n_steps)]
    time_labels_formatted = [time_to_hours(t) for t in time_labels]
    
    # Select representative time points for x-axis
    n_ticks = min(10, n_steps)
    tick_indices = np.linspace(0, n_steps-1, n_ticks, dtype=int)
    
    # === Plot 1: Average SHAP over time ===
    ax1 = fig.add_subplot(gs[0, :])
    ts_shap_avg = np.abs(ts_shap).mean(axis=0)
    
    ax1.plot(ts_shap_avg, linewidth=2, color='#ff0051')
    ax1.fill_between(range(len(ts_shap_avg)), ts_shap_avg, alpha=0.3, color='#ff0051')
    ax1.set_xlabel('Time', fontsize=12)
    ax1.set_ylabel('|SHAP Value|', fontsize=12)
    ax1.set_title(f'Time Series SHAP Importance Over Time (Sample {sample_idx}, Class {class_idx})', 
                  fontsize=14, fontweight='bold')
    ax1.set_xticks(tick_indices)
    ax1.set_xticklabels([time_labels_formatted[i] for i in tick_indices], rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # === Plot 2: SHAP heatmap with channel names ===
    ax2 = fig.add_subplot(gs[1, :])
    im = ax2.imshow(ts_shap, aspect='auto', cmap='RdBu_r', interpolation='nearest')
    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_ylabel('Channel', fontsize=12)
    ax2.set_title('Time Series SHAP Heatmap (All Channels)', fontsize=14, fontweight='bold')
    
    # Set channel labels
    if channel2feature is not None:
        channel_labels = [channel2feature.get(i, f'Ch{i}') for i in range(n_channels)]
        # Show subset of labels if too many
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
    
    # === Plot 3: Channel importance bar chart ===
    ax3 = fig.add_subplot(gs[2, :])
    channel_importance = np.abs(ts_shap).mean(axis=1)  # Average over time
    
    # Sort by importance
    sorted_idx = np.argsort(channel_importance)[::-1]
    sorted_importance = channel_importance[sorted_idx]
    
    if channel2feature is not None:
        sorted_names = [channel2feature.get(i, f'Ch{i}') for i in sorted_idx]
    else:
        sorted_names = [f'Channel {i}' for i in sorted_idx]
    
    # Show top 20
    n_show = min(20, len(sorted_importance))
    ax3.barh(range(n_show), sorted_importance[:n_show], color='#008bfb', alpha=0.7)
    ax3.set_yticks(range(n_show))
    ax3.set_yticklabels(sorted_names[:n_show], fontsize=9)
    ax3.set_xlabel('Mean |SHAP Value| Across Time', fontsize=12)
    ax3.set_title(f'Top {n_show} Most Important Channels', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')
    ax3.invert_yaxis()
    
    # === Plot 4: Categorical SHAP ===
    if shap_results['cat_shap'] is not None and shap_results['cat_shap'].size > 0:
        ax4 = fig.add_subplot(gs[3, 0])
        cat_shap = shap_results['cat_shap'][sample_idx]
        cat_data = shap_results['test_data']['cat'][sample_idx]
        
        if cat_shap.ndim == 2:
            cat_shap = cat_shap[:, class_idx]
        
        if feature_names_cat is None:
            feature_names_cat = [f'Cat_{i}' for i in range(len(cat_shap))]
        
        # Ensure feature names match the number of categorical features
        n_cat = len(cat_shap)
        if len(feature_names_cat) < n_cat:
            # Pad with generic names if not enough provided
            feature_names_cat = list(feature_names_cat) + [f'Cat_{i}' for i in range(len(feature_names_cat), n_cat)]
        elif len(feature_names_cat) > n_cat:
            # Truncate if too many provided
            feature_names_cat = feature_names_cat[:n_cat]
        
        colors = ['#ff0051' if x > 0 else '#008bfb' for x in cat_shap]
        ax4.barh(range(len(cat_shap)), cat_shap, color=colors, alpha=0.7)
        ax4.set_yticks(range(len(cat_shap)))
        ax4.set_yticklabels([f'{name}\n(val={int(val)})' 
                             for name, val in zip(feature_names_cat, cat_data)], fontsize=9)
        ax4.set_xlabel('SHAP Value', fontsize=12)
        ax4.set_title('Categorical Features', fontsize=14, fontweight='bold')
        ax4.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax4.grid(True, alpha=0.3, axis='x')
        ax4.invert_yaxis()
    
    # === Plot 5: Continuous SHAP ===
    if shap_results['cont_shap'] is not None and shap_results['cont_shap'].size > 0:
        ax5 = fig.add_subplot(gs[3, 1])
        cont_shap = shap_results['cont_shap'][sample_idx]
        cont_data = shap_results['test_data']['cont'][sample_idx]
        
        if cont_shap.ndim == 2:
            cont_shap = cont_shap[:, class_idx]
        
        if feature_names_cont is None:
            feature_names_cont = [f'Cont_{i}' for i in range(len(cont_shap))]
        
        # Ensure feature names match the number of continuous features
        n_cont = len(cont_shap)
        if len(feature_names_cont) < n_cont:
            # Pad with generic names if not enough provided
            feature_names_cont = list(feature_names_cont) + [f'Cont_{i}' for i in range(len(feature_names_cont), n_cont)]
        elif len(feature_names_cont) > n_cont:
            # Truncate if too many provided
            feature_names_cont = feature_names_cont[:n_cont]
        
        colors = ['#ff0051' if x > 0 else '#008bfb' for x in cont_shap]
        ax5.barh(range(len(cont_shap)), cont_shap, color=colors, alpha=0.7)
        ax5.set_yticks(range(len(cont_shap)))
        ax5.set_yticklabels([f'{name}\n(val={val:.2f})' 
                            for name, val in zip(feature_names_cont, cont_data)], fontsize=9)
        ax5.set_xlabel('SHAP Value', fontsize=12)
        ax5.set_title('Continuous Features', fontsize=14, fontweight='bold')
        ax5.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax5.grid(True, alpha=0.3, axis='x')
        ax5.invert_yaxis()
    
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
    Create summary visualizations for SHAP values across dataset with channel mapping.
    """
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3, height_ratios=[1, 1.2, 1])
    
    ts_shap = shap_results['ts_shap']
    
    # Handle multi-class output
    if ts_shap.ndim == 4:
        print(f"Multi-class output detected: {ts_shap.shape}, using class {class_idx}")
        ts_shap = ts_shap[:, :, :, class_idx]
    
    n_samples, n_channels, n_steps = ts_shap.shape
    
    # Create time labels
    time_labels = [step_to_time(i) for i in range(n_steps)]
    time_labels_formatted = [time_to_hours(t) for t in time_labels]
    n_ticks = min(10, n_steps)
    tick_indices = np.linspace(0, n_steps-1, n_ticks, dtype=int)
    
    # === Plot 1: Average importance over time ===
    ax1 = fig.add_subplot(gs[0, 0])
    ts_importance = np.abs(ts_shap).mean(axis=(0, 1))
    
    ax1.plot(ts_importance, linewidth=2, color='#ff0051')
    ax1.fill_between(range(len(ts_importance)), ts_importance, alpha=0.3, color='#ff0051')
    ax1.set_xlabel('Time', fontsize=12)
    ax1.set_ylabel('Mean |SHAP Value|', fontsize=12)
    ax1.set_title(f'Time Series Feature Importance Over Time (Class {class_idx})', 
                  fontsize=14, fontweight='bold')
    ax1.set_xticks(tick_indices)
    ax1.set_xticklabels([time_labels_formatted[i] for i in tick_indices], rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # === Plot 2: Channel importance bar chart ===
    ax2 = fig.add_subplot(gs[0, 1])
    channel_importance = np.abs(ts_shap).mean(axis=(0, 2))  # Average across samples and time
    
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
    ax2.set_title(f'Top {len(sorted_importance)} Most Important Channels', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.invert_yaxis()
    
    # === Plot 3: SHAP heatmap (channels x time) ===
    ax3 = fig.add_subplot(gs[1, :])
    # Average SHAP values across samples
    ts_shap_mean = np.abs(ts_shap).mean(axis=0)  # [n_channels, n_steps]
    
    im = ax3.imshow(ts_shap_mean, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    ax3.set_xlabel('Time', fontsize=12)
    ax3.set_ylabel('Channel', fontsize=12)
    ax3.set_title('Time Series SHAP Heatmap (Mean Across Cohort)', fontsize=14, fontweight='bold')
    
    # Set channel labels
    if channel2feature is not None:
        channel_labels = [channel2feature.get(i, f'Ch{i}') for i in range(n_channels)]
        if n_channels > 25:
            tick_step = max(1, n_channels // 25)
            y_ticks = list(range(0, n_channels, tick_step))
            y_labels = [channel_labels[i] for i in y_ticks]
            ax3.set_yticks(y_ticks)
            ax3.set_yticklabels(y_labels, fontsize=8)
        else:
            ax3.set_yticks(range(n_channels))
            ax3.set_yticklabels(channel_labels, fontsize=9)
    
    ax3.set_xticks(tick_indices)
    ax3.set_xticklabels([time_labels_formatted[i] for i in tick_indices], rotation=45)
    plt.colorbar(im, ax=ax3, label='Mean |SHAP Value|')
    
    # === Plot 4: Categorical features ===
    if shap_results['cat_shap'] is not None and shap_results['cat_shap'].size > 0:
        ax4 = fig.add_subplot(gs[2, 0])
        cat_shap = shap_results['cat_shap']
        
        if cat_shap.ndim == 3:
            cat_shap = cat_shap[:, :, class_idx]
        
        cat_importance = np.abs(cat_shap).mean(axis=0) if cat_shap.ndim > 1 else np.abs(cat_shap)
        
        if feature_names_cat is None:
            feature_names_cat = [f'Cat_{i}' for i in range(len(cat_importance))]
        
        n_features = min(len(cat_importance), len(feature_names_cat))
        cat_importance = cat_importance[:n_features]
        feature_names_cat = feature_names_cat[:n_features]
        
        sorted_idx = np.argsort(cat_importance)[::-1][:max_display]
        sorted_importance = cat_importance[sorted_idx]
        sorted_names = [feature_names_cat[int(i)] for i in sorted_idx]
        
        ax4.barh(range(len(sorted_importance)), sorted_importance, color='#ff0051', alpha=0.7)
        ax4.set_yticks(range(len(sorted_importance)))
        ax4.set_yticklabels(sorted_names, fontsize=10)
        ax4.set_xlabel('Mean |SHAP Value|', fontsize=12)
        ax4.set_title(f'Top {len(sorted_importance)} Categorical Features', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='x')
        ax4.invert_yaxis()
    else:
        ax4 = fig.add_subplot(gs[2, 0])
        ax4.text(0.5, 0.5, 'No categorical features', ha='center', va='center', fontsize=14)
        ax4.axis('off')
    
    # === Plot 5: Continuous features ===
    if shap_results['cont_shap'] is not None and shap_results['cont_shap'].size > 0:
        ax5 = fig.add_subplot(gs[2, 1])
        cont_shap = shap_results['cont_shap']
        
        if cont_shap.ndim == 3:
            cont_shap = cont_shap[:, :, class_idx]
        
        cont_importance = np.abs(cont_shap).mean(axis=0) if cont_shap.ndim > 1 else np.abs(cont_shap)
        
        if feature_names_cont is None:
            feature_names_cont = [f'Cont_{i}' for i in range(len(cont_importance))]
        
        n_features = min(len(cont_importance), len(feature_names_cont))
        cont_importance = cont_importance[:n_features]
        feature_names_cont = feature_names_cont[:n_features]
        
        sorted_idx = np.argsort(cont_importance)[::-1][:max_display]
        sorted_importance = cont_importance[sorted_idx]
        sorted_names = [feature_names_cont[int(i)] for i in sorted_idx]
        
        ax5.barh(range(len(sorted_importance)), sorted_importance, color='#008bfb', alpha=0.7)
        ax5.set_yticks(range(len(sorted_importance)))
        ax5.set_yticklabels(sorted_names, fontsize=10)
        ax5.set_xlabel('Mean |SHAP Value|', fontsize=12)
        ax5.set_title(f'Top {len(sorted_importance)} Continuous Features', fontsize=14, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='x')
        ax5.invert_yaxis()
    else:
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.text(0.5, 0.5, 'No continuous features', ha='center', va='center', fontsize=14)
        ax5.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def prepare_learner(data):
    MODEL_NAME = '10122025'

    mixed_dls = data["mixed_dls"]
    holdout_mixed_dls = data["holdout_mixed_dls"]
    num_cols = data["num_cols"]
    classes = data["classes"]

    # Load model
    backbone = TSTabFusionTransformer(
        c_in=mixed_dls.vars,
        c_out=2,
        seq_len=mixed_dls.len,
        classes=classes,
        cont_names=num_cols,
        n_layers=8,
        n_heads=8,
        d_model=64,
        fc_dropout=0.75,
        res_dropout=0.22,
        fc_mults=(0.3, 0.1),
    )
    loss_func = LabelSmoothingCrossEntropyFlat()
    learn = Learner(mixed_dls, backbone, loss_func=loss_func, metrics=None)
    learn = learn.load(MODEL_NAME)

    return learn

# ============================================================================
def main():
    """
    get data, calculate shap, visualize both cohort and example individual for retrospective data
    TODO: get model cfg from cfg file
    """
    data = prepare_data_and_dls()
    learn= prepare_learner(data)

    shap_results = calculate_shap_from_dataloaders(
                                model=learn.model,
                                background_loader=data["mixed_dls"],
                                test_loader=data["holdout_mixed_dls"],
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
        save_path='reports/shap_summary_cohort.png'
    )

    # Visualize for a specific observation
    visualize_shap_individual(
        shap_results,
        sample_idx=1,
        channel2feature=channel2feature,
        feature_names_cat=cfg["dataset"]["cat_cols"],
        feature_names_cont=cfg["dataset"]["num_cols"],
        class_idx=1,
        save_path='reports/shap_individual_sample_1.png'
    )
        
