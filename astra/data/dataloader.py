import pandas as pd
import numpy as np

from fastai.data.transforms import Categorize
from fastai.tabular.core import Categorify, FillMissing, Normalize

from tsai.data.core import get_ts_dls
from tsai.data.preprocessing import TSStandardize
from tsai.data.tabular import get_tabular_dls
from tsai.data.mixed import get_mixed_dls
from tsai.data.preparation import df2xy

from astra.utils import get_base_df, logger, align_dataframes
from astra.data.preprocessing import MultiHotCategoricalEncoder
from astra.data.datasets import TSDS 


### Utility functions
def tscatdfwide2x(df_wide:pd.DataFrame, sample_col:str='PID', cat_col='FEATURE'):
    encoder = MultiHotCategoricalEncoder()
    X_multi_hot, encoding_info = encoder.fit_transform(
        df_wide,
        sample_col=sample_col,
        timestep_cols=df_wide.timestep_cols,
        cat_col=cat_col,
        feature_names=df_wide.FEATURE.dropna().unique()

    )
    return X_multi_hot, encoding_info 

def dfwide2ts_dls(df_wide, y, cfg, encoder=None):
    """
    Create categorical TS dataloader with optional pre-fitted encoder.
    
    Args:
        df_wide: Wide-format DataFrame with categorical TS
        y: Targets
        cfg: Config dict
        encoder: Optional pre-fitted MultiHotCategoricalEncoder
                 If None, will fit new encoder
                 If provided, will use for transform only
    
    Returns:
        Tuple of (ts_cat_dls, encoding_info, encoder)
    """
    if encoder is None:
        # Fit new encoder
        encoder = MultiHotCategoricalEncoder()
        X_multi_hot, encoding_info = encoder.fit_transform(
            df_wide,
            sample_col='PID',
            timestep_cols=df_wide.timestep_cols,
            cat_col='FEATURE',
            feature_names=df_wide.FEATURE.dropna().unique()
        )
    else:
        # Use pre-fitted encoder (for holdout)
        X_multi_hot, encoding_info = encoder.transform(
            df_wide,
            sample_col='PID',
            timestep_cols=df_wide.timestep_cols,
            cat_col='FEATURE'
        )
    
    print(f"X_multi_hot shape: {X_multi_hot.shape}")
    
    ts_cat_dls = get_ts_dls(
        X_multi_hot.astype(np.int64), 
        y, 
        splits=None, 
        bs=cfg["training"]["bs"]
    )
    
    # Add cat dimensions from encoding info for model init later
    ts_cat_dims = {
        feat_name: end - start 
        for feat_name, (start, end) in encoding_info['feature_ranges'].items()
    }
    ts_cat_dls.ts_cat_dims = ts_cat_dims
    ts_cat_dls.X_multi_hot = X_multi_hot
    return ts_cat_dls, encoding_info, encoder

### Default data prep to dls

def prepare_data_and_dls(cfg):
    """
    Prepare data and dataloaders with proper encoding_info tracking.
    
    FIXED:
    - Fit encoder on trainval, apply to holdout
    - Track encoding_info for both splits
    - Return encoding_info in data dict
    - Fix duplicate "X" key in return dict
    """
    # Load dataframes
    base = get_base_df()
    if cfg["dataset"]["exclusion"] == "lvl1tc":
        base = base[base.LVL1TC == 1]

    concepts = cfg["concepts"]
    
    holdout = TSDS(cfg, base[base.ServiceDate > '06-01-2023'].copy(deep=True))
    trainval = TSDS(cfg, base[base.ServiceDate <= '06-01-2023'].copy(deep=True))

    # Split concepts into categorical and continuous
    for tsds in [holdout, trainval]:
        tsds.cat_concepts = {
            k: tsds.concepts[k] 
            for k in cfg["dataset"]["ts_cat_names"] 
            if k in tsds.concepts
        }
        tsds.cont_concepts = {
            k: v 
            for k, v in tsds.concepts.items() 
            if k not in cfg["dataset"]["ts_cat_names"]
        }
        tsds.complete = pd.concat(tsds.cont_concepts).fillna(0.0)
        tsds.complete_cat = pd.concat(tsds.cat_concepts)
        tsds.complete_cat.timestep_cols = tsds.timestep_cols
    
    # Align dataframes
    trainval.complete, holdout.complete = align_dataframes(
        trainval.complete, 
        holdout.complete
    )
    
    cat_cols = cfg["dataset"]["cat_cols"]
    num_cols = cfg["dataset"]["num_cols"]
    logger.info(f'Categoricals: {cat_cols}\nNumericals: {num_cols}')

    # Common transforms
    tfms = [None, [Categorize()]]
    batch_tfms = TSStandardize(by_var=True)
    procs = [Categorify, FillMissing, Normalize]

    # Get classes from combined data
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

    # ============================================================================
    # TRAINVAL DATALOADERS
    # ============================================================================
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

    # FIT encoder on trainval
    ts_cat_dls, encoding_info, cat_encoder = dfwide2ts_dls(
        trainval.complete_cat, 
        y, 
        cfg,
        encoder=None  # Fit new encoder
    )
    
    mixed_dls = get_mixed_dls(
        ts_dls,
        tab_dls,
        ts_cat_dls,
        bs=cfg["training"]["bs"]
    )

    # ============================================================================
    # HOLDOUT DATALOADERS
    # ============================================================================
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
    
    # TRANSFORM holdout using fitted encoder (CRITICAL!)
    test_ts_cat_dls, holdout_encoding_info, _ = dfwide2ts_dls(
        holdout.complete_cat, 
        ty, 
        cfg,
        encoder=cat_encoder  # Use fitted encoder from trainval
    )

    holdout_mixed_dls = get_mixed_dls(
        test_ts_dls,
        test_tab_dls,
        test_ts_cat_dls,
        bs=cfg["training"]["bs"]
    )

    # ============================================================================
    # RETURN DATA DICT
    # ============================================================================
    return {
        "base": base,
        "trainval": trainval,
        "holdout": holdout,
        "X": X,
        "X_multi_hot": ts_cat_dls.X_multi_hot, 
        "y": y,
        "tX": tX,
        "tX_multi_hot":test_ts_cat_dls.X_multi_hot,
        "ty":ty,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "tfms": tfms,
        "batch_tfms": batch_tfms,
        "procs": procs,
        "classes": classes,
        "mixed_dls": mixed_dls,
        "holdout_mixed_dls": holdout_mixed_dls,
        "ts_dls": ts_dls,
        "holdout_ts_dls": test_ts_dls,
        "ts_cat_dls": ts_cat_dls,
        "holdout_ts_cat_dls": test_ts_cat_dls,
        "encoding_info": encoding_info,  # For SHAP and model init
        "cat_encoder": cat_encoder,  # For future transforms
    }


