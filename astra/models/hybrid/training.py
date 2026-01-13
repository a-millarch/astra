import torch
from fastai.learner import Learner
from tqdm.auto import tqdm


import argparse
import os
import copy
import warnings
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import base64
import io
from scipy import stats
import pickle

from omegaconf import OmegaConf

from tsai.data.core import get_ts_dls
from tsai.data.preprocessing import TSStandardize
from tsai.data.tabular import get_tabular_dls
from tsai.data.mixed import get_mixed_dls
from tsai.data.validation import get_splits
from tsai.data.preparation import df2xy
from tsai.all import TSTabFusionTransformer, LabelSmoothingCrossEntropyFlat, Learner

#from fastai.callback.core import TrainEvalCallback, CancelValidException
from fastai.data.transforms import Categorize
from fastai.tabular.core import Categorify, FillMissing, Normalize
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve

from astra.utils import cfg, logger, get_base_df, align_dataframes, clear_mem, get_cfg
from astra.data.datasets import TSDS
from astra.models.hybrid.mlm import TSTabFusionMLM, MLMConfig, pretrain_mlm_enhanced

#from astra.visualize.evaluation import plot_evaluation, plot_multiple_evaluations, plot_time_metrics

from astra.evaluation.hybrid_model import run_eval
from astra.models.callbacks import SkipValidationCallback



def run_pretrain(data, pretrain_cfg=None, device='cuda'):
    """
    data: dict returned by prepare_data_and_dls()
    pretrain_cfg: MLMConfig or None (then a default is used)
    """
    X, y = data["X"], data["y"]
    num_cols = data["num_cols"]
    classes = data["classes"]
    tfms = data["tfms"]
    batch_tfms = data["batch_tfms"]
    procs = data["procs"]

    # Create splits for unlabeled pretraining (train/valid on same X,y)
    ts_splits = get_splits(
        y,
        valid_size=0.2,
        stratify=True,
        random_state=42,
        shuffle=True,
        check_splits=True,
        show_plot=True
    )

    splits = (ts_splits[0], ts_splits[1])
    #splits = ((ts_splits[0],)+ (ts_splits[1],))
    
    ts_dls_ul = get_ts_dls(
        X,
        splits=splits,
        tfms=tfms,
        batch_tfms=batch_tfms,
        bs=cfg["training"]["bs"],
        drop_last=False
    )

    tab_dls_ul = get_tabular_dls(
        data["trainval"].tab_df,
        procs=procs,
        cat_names=data["cat_cols"].copy(),
        cont_names=num_cols.copy(),
        splits=splits,
        bs=cfg["training"]["bs"],
        drop_last=False
    )

    mixed_dls_ul = get_mixed_dls(
        ts_dls_ul,
        tab_dls_ul,
        bs=cfg["training"]["bs"]
    )

    backbone = TSTabFusionTransformer(
        c_in=mixed_dls_ul.vars,
        c_out=2,
        seq_len=mixed_dls_ul.len,
        classes=classes,
        cont_names=num_cols,
        n_layers=8,
        n_heads=8,
        d_model=64,
        fc_dropout=0.75,
        res_dropout=0.22,
        fc_mults=(0.3, 0.1),
    )

    if pretrain_cfg is None:
        pretrain_cfg = MLMConfig(
            mask_prob_ts=0.10,
            mask_prob_cat=0.15,
            mask_prob_cont=0.15,
            epochs=50,
            lr=1e-5,
            warmup_epochs=3,
            ts_loss_weight=1.0,
            cat_loss_weight=1.0,
            cont_loss_weight=1.0,
            contrastive_weight=1.0,
            patience=10,
            save_best=True,
            checkpoint_dir='./pretrain_checkpoints'
        )

    mlm_model = TSTabFusionMLM(backbone, pretrain_cfg)

    history = pretrain_mlm_enhanced(
        mlm_model,
        train_loader=mixed_dls_ul.train,
        val_loader=mixed_dls_ul.valid,
        config=pretrain_cfg,
        device=device
    )

    logger.info('Pretrained model saved')
    return pretrain_cfg, mlm_model, mixed_dls_ul


def run_finetune(
    data,
    model_name: str,
    use_pretrained: bool = True,
    pretrain_cfg: MLMConfig = None,
    skip_valid: bool = True,
    lr: float = 4.7863e-4,
    n_epochs: int = 22
):
    """
    Fine-tune classifier.
    If use_pretrained=True, loads backbone from pretrain checkpoint.
    """
    mixed_dls = data["mixed_dls"]
    num_cols = data["num_cols"]
    classes = data["classes"]

    # Build backbone
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

    if use_pretrained:
        if pretrain_cfg is None:
            checkpoint_dir = './pretrain_checkpoints'
        else:
            checkpoint_dir = pretrain_cfg.checkpoint_dir
        checkpoint = torch.load(os.path.join(checkpoint_dir, 'best_model.pt'))
        # need the same MLM model structure to load, then extract backbone
        mlm_model = TSTabFusionMLM(backbone, pretrain_cfg)
        mlm_model.load_state_dict(checkpoint['model_state_dict'])
        backbone = mlm_model.backbone

    loss_func = LabelSmoothingCrossEntropyFlat()
    cbs = [SkipValidationCallback()] if skip_valid else None

    learn = Learner(
        mixed_dls,
        backbone,
        loss_func=loss_func,
        metrics=None,
        cbs=cbs
    )

    learn.fit_one_cycle(n_epochs, lr)
    learn.save(model_name)
    clear_mem()
    return learn


def get_preds_mixed(learner, dl, with_input=False, with_decoded=False, 
                     with_loss=False, act=None, save_preds=None, save_targs=None,
                     concat_dim=0, debug=False):
    """
    Custom get_preds that works with TSAI mixed dataloaders.
    
    Drop-in replacement for learner.get_preds() that bypasses the
    problematic callback system.
    
    Args:
        learner: fastai Learner
        dl: DataLoader (e.g., mixed_dls.train)
        with_input: Return inputs
        with_decoded: Return decoded predictions
        with_loss: Return losses
        act: Activation function to apply (default: from loss_func)
        save_preds: Path to save predictions
        save_targs: Path to save targets
        concat_dim: Dimension to concatenate results
    
    Returns:
        Tuple of (predictions, targets) or more depending on flags
    """
    learner.model.eval()
    
    all_preds = []
    all_targets = []
    all_inputs = [] if with_input else None
    all_losses = [] if with_loss else None
    
    # Get activation function
    if act is None:
        act = getattr(learner.loss_func, 'activation', None)
        if act is None:
            act = torch.nn.Identity()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dl, desc="Getting predictions", leave=False)):
            if debug and batch_idx == 0:
                print(f"\n=== DEBUG: First Batch ===")
                print(f"Batch type: {type(batch)}")
                print(f"Batch length: {len(batch) if isinstance(batch, (tuple, list)) else 'N/A'}")
                if isinstance(batch, (tuple, list)):
                    for i, elem in enumerate(batch):
                        print(f"  batch[{i}]: type={type(elem)}, ", end="")
                        if torch.is_tensor(elem):
                            print(f"shape={elem.shape}, dtype={elem.dtype}")
                        elif isinstance(elem, (tuple, list)):
                            print(f"length={len(elem)}")
                            for j, sub in enumerate(elem):
                                print(f"    batch[{i}][{j}]: type={type(sub)}, ", end="")
                                if torch.is_tensor(sub):
                                    print(f"shape={sub.shape}")
                                else:
                                    print(f"value={sub}")
                        else:
                            print(f"value={elem}")
                print("=========================\n")
            
            # Unpack batch - handle different formats
            if isinstance(batch, (tuple, list)):
                if len(batch) == 2:
                    inputs, targets = batch
                elif len(batch) == 1:
                    # Only inputs, no targets
                    inputs = batch[0]
                    targets = None
                else:
                    raise ValueError(f"Unexpected batch format with {len(batch)} elements")
            else:
                # Single element batch
                inputs = batch
                targets = None
            
            # Move to device
            device = learner.dls.device if hasattr(learner.dls, 'device') else 'cuda'
            
            if isinstance(inputs, (tuple, list)):
                # Mixed dataloader: (x_ts, x_tab, x_ts_cat)
                inputs_device = []
                for inp in inputs:
                    if isinstance(inp, (tuple, list)):
                        # Nested tuple (x_tab)
                        inputs_device.append(tuple(i.to(device) if torch.is_tensor(i) else i for i in inp))
                    elif torch.is_tensor(inp):
                        inputs_device.append(inp.to(device))
                    else:
                        inputs_device.append(inp)
                inputs_device = tuple(inputs_device)
            else:
                inputs_device = inputs.to(device) if torch.is_tensor(inputs) else inputs
            
            # Handle targets
            if targets is not None:
                # Handle case where targets might be a tuple or list
                if isinstance(targets, (tuple, list)):
                    targets = targets[0] if len(targets) > 0 else targets
                
                # Ensure targets is a tensor
                if not torch.is_tensor(targets):
                    targets = torch.tensor(targets, device=device)
                elif targets.device != torch.device(device):
                    targets = targets.to(device)
            else:
                # No targets in batch - create dummy targets
                # This shouldn't happen with get_preds, but handle it gracefully
                batch_size = inputs_device[0].shape[0] if isinstance(inputs_device, tuple) else inputs_device.shape[0]
                targets = torch.zeros(batch_size, dtype=torch.long, device=device)
            
            # Forward pass
            preds = learner.model(inputs_device)
            
            # Apply activation
            if act is not None:
                preds = act(preds)
            
            # Calculate loss if requested
            if with_loss:
                loss = learner.loss_func(preds, targets)
                all_losses.append(loss.detach().cpu())
            
            # Store results
            all_preds.append(preds.detach().cpu())
            all_targets.append(targets.detach().cpu())
            
            if with_input:
                # Store inputs (this can be memory intensive)
                if isinstance(inputs_device, (tuple, list)):
                    inputs_cpu = []
                    for inp in inputs_device:
                        if isinstance(inp, (tuple, list)):
                            inputs_cpu.append(tuple(i.cpu() if torch.is_tensor(i) else i for i in inp))
                        elif torch.is_tensor(inp):
                            inputs_cpu.append(inp.cpu())
                        else:
                            inputs_cpu.append(inp)
                    all_inputs.append(tuple(inputs_cpu))
                else:
                    all_inputs.append(inputs_device.cpu())
    
    # Concatenate results
    preds = torch.cat(all_preds, dim=concat_dim)
    targets = torch.cat(all_targets, dim=concat_dim)
    
    # Save if requested
    if save_preds is not None:
        torch.save(preds, save_preds)
    if save_targs is not None:
        torch.save(targets, save_targs)
    
    # Build return tuple
    result = [preds, targets]
    
    if with_input:
        # Inputs are harder to concatenate due to nested structure
        result = [all_inputs] + result
    
    if with_loss:
        losses = torch.stack(all_losses)
        result.append(losses)
    
    if with_decoded:
        # For classification, decoded = argmax
        if preds.ndim > 1 and preds.shape[1] > 1:
            decoded = torch.argmax(preds, dim=1)
        else:
            decoded = preds
        result.insert(1, decoded)
    
    return tuple(result) if len(result) > 2 else (result[0], result[1])


def patch_learner_get_preds(learner):
    """
    Patch a Learner instance to use the custom get_preds.
    
    Args:
        learner: fastai Learner instance
    
    Returns:
        Same learner (modified in-place)
    
    Usage:
        learn = Learner(mixed_dls, backbone, ...)
        learn = patch_learner_get_preds(learn)
        
        # Now this works!
        preds, targs = learn.get_preds(dl=mixed_dls.train)
    """
    import types
    
    def custom_get_preds(self, ds_idx=1, dl=None, with_input=False, with_decoded=False,
                         with_loss=False, act=None, inner=False, reorder=True, 
                         cbs=None, save_preds=None, save_targs=None, concat_dim=0):
        """Custom get_preds that bypasses callbacks."""
        if dl is None:
            dl = self.dls[ds_idx]
        
        return get_preds_mixed(
            self, dl, 
            with_input=with_input,
            with_decoded=with_decoded,
            with_loss=with_loss,
            act=act,
            save_preds=save_preds,
            save_targs=save_targs,
            concat_dim=concat_dim
        )
    
    # Replace get_preds method
    learner.get_preds = types.MethodType(custom_get_preds, learner)
    
    return learner


# ============================================================================
# Convenience Function
# ============================================================================

def create_learner_with_working_get_preds(dls, model, loss_func=None, opt_func=None, 
                                          lr=0.001, splitter=None, cbs=None, 
                                          metrics=None, path=None, model_dir='models', 
                                          wd=None, wd_bn_bias=False, train_bn=True, 
                                          moms=(0.95, 0.85, 0.95)):
    """
    Create a Learner with working get_preds() for mixed dataloaders.
    
    Drop-in replacement for Learner() that automatically patches get_preds.
    
    Usage:
        learn = create_learner_with_working_get_preds(
            mixed_dls, backbone,
            loss_func=nn.CrossEntropyLoss(),
            metrics=[accuracy]
        )
        
        # Everything works!
        learn.fit_one_cycle(22, lr=4.7863e-4)
        preds, targs = learn.get_preds(dl=mixed_dls.train)
    """
    learn = Learner(
        dls, model, 
        loss_func=loss_func,
        opt_func=opt_func,
        lr=lr,
        splitter=splitter,
        cbs=cbs,
        metrics=metrics,
        path=path,
        model_dir=model_dir,
        wd=wd,
        wd_bn_bias=wd_bn_bias,
        train_bn=train_bn,
        moms=moms
    )
    
    # Patch get_preds
    learn = patch_learner_get_preds(learn)
    
    return learn

