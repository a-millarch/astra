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

# -----------------------------
# 1) Data loading and setup
# -----------------------------

def prepare_data_and_dls():
    # Load dataframes
    base = get_base_df()
    if cfg["dataset"]["exclusion"] == "lvl1tc":
        base = base[base.LVL1TC ==1]
    tsds = TSDS(cfg, base)

    concepts = cfg["concepts"]
    #['VitaleVaerdier', 'Labsvar', 'Medicin', 'ITAOversigtsrapport']

    holdout = TSDS(cfg, base[base.ServiceDate >= '06-01-2023'].copy(deep=True))
    trainval = TSDS(cfg, base[base.ServiceDate <= '06-01-2023'].copy(deep=True))

    trainval.complete = pd.concat(trainval.concepts).fillna(0.0)
    holdout.complete = pd.concat(holdout.concepts).fillna(0.0)
    trainval.complete, holdout.complete = align_dataframes(trainval.complete, holdout.complete)

    cat_cols = cfg["dataset"]["cat_cols"]
    num_cols = cfg["dataset"]["num_cols"]
    logger.info(f'Categoricals: {cat_cols}\nNumericals: {num_cols}')

    # Common transforms
    tfms = [None, [Categorize()]]
    batch_tfms = TSStandardize(by_var=True)
    procs = [Categorify, FillMissing, Normalize]

    # ---------------- holdout dls ----------------
    logger.info('Preparing holdout dataloader')
    tX, ty = df2xy(
        holdout.complete,
        sample_col='PID',
        feat_col='FEATURE',
        data_cols=holdout.complete.columns[3:-1],
        target_col=holdout.target
    )
    ty = list(ty[:, 0].flatten())
    logger.info(f'Holdout X shape: {tX.shape}')

    test_ts_dls = get_ts_dls(
        tX, ty,
        splits=None,
        tfms=tfms,
        batch_tfms=batch_tfms,
        bs=cfg["training"]["bs"],
        drop_last=False,
        shuffle=False
    )

    test_tab_dls = get_tabular_dls(
        holdout.tab_df,
        procs=procs,
        cat_names=cat_cols.copy(),
        cont_names=num_cols.copy(),
        y_names=cfg["target"],
        splits=None,
        drop_last=False,
        shuffle=False
    )

    holdout_mixed_dls = get_mixed_dls(
        test_ts_dls,
        test_tab_dls,
        bs=cfg["training"]["bs"],
        shuffle_valid=False
    )

    # For classes
    complete_tab_dls = get_tabular_dls(
        pd.concat([trainval.tab_df, holdout.tab_df]),
        procs=procs,
        cat_names=cat_cols.copy(),
        cont_names=num_cols.copy(),
        y_names=cfg["target"],
        splits=None,
        drop_last=False,
        shuffle=False
    )
    classes = complete_tab_dls.classes

    # ---------------- train/val dls ----------------
    logger.info("Setting up X,y for training and validation")
    X, y = df2xy(
        trainval.complete,
        sample_col='PID',
        feat_col='FEATURE',
        data_cols=trainval.complete.columns[3:-1],
        target_col=cfg["target"]
    )
    y = list(y[:, 0].flatten())
    logger.info(f'Train/val X shape: {X.shape}')

    ts_dls = get_ts_dls(
        X, y,
        splits=None,
        tfms=tfms,
        batch_tfms=batch_tfms,
        bs=cfg["training"]["bs"],
        drop_last=False
    )

    tab_dls = get_tabular_dls(
        trainval.tab_df,
        procs=procs,
        cat_names=cat_cols.copy(),
        cont_names=num_cols.copy(),
        y_names=cfg["target"],
        splits=None,
        bs=cfg["training"]["bs"],
        drop_last=False
    )

    mixed_dls = get_mixed_dls(
        ts_dls,
        tab_dls,
        bs=cfg["training"]["bs"]
    )

    return {
        "base": base,
        "trainval": trainval,
        "holdout": holdout,
        "X": X,
        "y": y,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "tfms": tfms,
        "batch_tfms": batch_tfms,
        "procs": procs,
        "classes": classes,
        "mixed_dls": mixed_dls,
        "holdout_mixed_dls": holdout_mixed_dls,
        "ts_dls":ts_dls,
        "holdout_ts_dls":test_ts_dls
    }


# -----------------------------
# 2) Pretraining
# -----------------------------

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


# -----------------------------
# 3) Fine-tuning
# -----------------------------


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


# -----------------------------
# 4) Evaluation
# -----------------------------


def parse_args():
    parser = argparse.ArgumentParser(description='MLM Pretraining + Fine-tuning Pipeline')
    
    parser.add_argument('--model-name', '-m', default='10122025', help='Model name')
    parser.add_argument('--lr', type=float, default=4.7863e-4, help='Learning rate')
    parser.add_argument('--finetune-epochs', type=int, default=22, help='Fine-tune epochs')
    
    # Stage flags
    parser.add_argument('--pretrain', action='store_true', default=True)
    parser.add_argument('--no-pretrain', dest='pretrain', action='store_false')
    parser.add_argument('--finetune', action='store_true', default=True)
    parser.add_argument('--no-finetune', dest='finetune', action='store_false')
    parser.add_argument('--eval', action='store_true', default=True)
    parser.add_argument('--no-eval', dest='eval', action='store_false')
    
    # New eval flag
    parser.add_argument('--comprehensive-eval', action='store_true', default=True,
                       help='Run full time-dependent evaluation (default)')
    parser.add_argument('--simple-eval', dest='comprehensive_eval', action='store_false',
                       help='Run simple single-point evaluation')
    
    parser.add_argument('--use-pretrained', action='store_true', default=True)
    parser.add_argument('--skip-valid', action='store_true', default=True)
    parser.add_argument('--device', default='cuda')
    
    return parser.parse_args()

def main():
    args = parse_args()
    data = prepare_data_and_dls()
    pretrain_cfg = None
    
    if args.pretrain:
        logger.info("=== Running Pretraining ===")
        pretrain_cfg, _, _ = run_pretrain(data, device=args.device)
    
    if args.finetune:
        logger.info("=== Running Fine-tuning ===")
        run_finetune(data, args.model_name, args.use_pretrained, pretrain_cfg,
                    args.skip_valid, args.lr, args.finetune_epochs)
    
    if args.eval:
        logger.info("=== Running Evaluation ===")
        results = run_eval(data, args.model_name, args.comprehensive_eval)
        if args.comprehensive_eval:
            logger.info(f"Generated {len(results[0])} time points with CIs")

if __name__ == "__main__":
    main()