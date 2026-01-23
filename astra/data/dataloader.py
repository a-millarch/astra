import pandas as pd
import numpy as np

from fastai.data.transforms import Categorize
from fastai.tabular.core import Categorify, FillMissing, Normalize

from tsai.data.core import get_ts_dls
from tsai.data.preprocessing import TSStandardize
from tsai.data.tabular import get_tabular_dls
from tsai.data.mixed import get_mixed_dls
from tsai.data.preparation import df2xy

from sklearn.preprocessing import RobustScaler, StandardScaler
import pickle

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
    
    logger.debug(f"X_multi_hot shape: {X_multi_hot.shape}")
    
    ts_cat_dls = get_ts_dls(
        X_multi_hot.astype(np.int64), 
        y, 
        splits=None, 
        bs=cfg["training"]["bs"],
        shuffle=False
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
    Prepare data and dataloaders with FIXED normalization.
    
    CRITICAL CHANGES:
    - Fit normalization on trainval ONLY (no holdout data leakage)
    - Apply same normalization to holdout
    - No batch-level transforms (reproducible results)
    - Store scalers for deployment
    
    FIXED (from before):
    - Fit encoder on trainval, apply to holdout
    - Track encoding_info for both splits
    - Return encoding_info in data dict
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

    # Common transforms (NO batch transforms!)
    tfms = [None, [Categorize()]]
    batch_tfms = None  # ← REMOVED! No more batch-level standardization
    procs = [Categorify, FillMissing]  # ← REMOVED Normalize!
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
    # TRAINVAL DATA EXTRACTION
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
    logger.info(f'Train/val X shape (before normalization): {X.shape}')

    # ============================================================================
    # FIT SCALERS ON TRAINVAL ONLY (NO DATA LEAKAGE!)
    # ============================================================================
    logger.info("Fitting normalization scalers on trainval data only...")
    
    # 1. CONTINUOUS TIME SERIES SCALER
    # Use StandardScaler for data with many zeros (better than RobustScaler)
    ts_scaler = StandardScaler()  # Uses mean/std (works with sparse data)
    
    # Reshape: [samples, seq_len, n_features] → [samples*seq_len, n_features]
    X_reshaped = X.reshape(-1, X.shape[2])
    logger.info(f'Fitting TS scaler on {X_reshaped.shape[0]} timesteps, {X_reshaped.shape[1]} features')
    
    # FIT on trainval only
    ts_scaler.fit(X_reshaped)
    
    # TRANSFORM trainval
    X_normalized = ts_scaler.transform(X_reshaped).reshape(X.shape)
    logger.info(f'Train/val X shape (after normalization): {X_normalized.shape}')
    logger.info(f'Train/val X normalized stats: mean={X_normalized.mean():.4f}, std={X_normalized.std():.4f}')
    
    # 2. TABULAR DATA SCALER
    tab_scaler = StandardScaler()  # Also use StandardScaler for tabular
    
    # Only normalize continuous columns
    if num_cols:
        logger.info(f'Fitting tabular scaler on {len(num_cols)} continuous features')
        
        # FIT on trainval only
        tab_scaler.fit(trainval.tab_df[num_cols])
        
        # TRANSFORM trainval
        trainval_tab_normalized = trainval.tab_df.copy()
        trainval_tab_normalized[num_cols] = tab_scaler.transform(trainval.tab_df[num_cols])
        
        logger.info(f'Tabular normalized stats: mean={trainval_tab_normalized[num_cols].mean().mean():.4f}, '
                   f'std={trainval_tab_normalized[num_cols].std().mean():.4f}')
    else:
        trainval_tab_normalized = trainval.tab_df
        logger.info('No continuous tabular features to normalize')

    # ============================================================================
    # TRAINVAL DATALOADERS (with normalized data, no batch transforms)
    # ============================================================================
    logger.info("Creating trainval dataloaders with fixed normalization...")
    
    ts_dls = get_ts_dls(
        X_normalized,  # ← Pre-normalized!
        y,
        splits=None,
        tfms=tfms,
        batch_tfms=None,  # ← NO batch transforms!
        bs=cfg["training"]["bs"],
        drop_last=False,
          shuffle=False
    )
    logger.info(f'3 {cfg["target"]}')
    
    
    tab_dls = get_tabular_dls(
        trainval_tab_normalized,  # ← Pre-normalized!
        procs=procs,  # ← Normalize removed!
        cat_names=cat_cols.copy(),
        cont_names=num_cols.copy(),
        y_names=cfg["target"],
        splits=None,
        bs=cfg["training"]["bs"],
        drop_last=False,
          shuffle=False
    )

    # FIT categorical encoder on trainval
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
    # HOLDOUT DATA EXTRACTION
    # ============================================================================
    logger.info('Preparing holdout data...')
    tX, ty = df2xy(
        holdout.complete,
        sample_col='PID',
        feat_col='FEATURE',
        data_cols=holdout.complete.columns[3:-1],
        target_col=holdout.target
    )
    ty = list(ty[:, 0].flatten())
    logger.info(f'Holdout X shape (before normalization): {tX.shape}')

    # ============================================================================
    # TRANSFORM HOLDOUT WITH TRAINVAL SCALERS (NO FITTING!)
    # ============================================================================
    logger.info("Applying trainval normalization to holdout (no data leakage)...")
    
    # 1. TRANSFORM continuous TS with FITTED scaler
    tX_reshaped = tX.reshape(-1, tX.shape[2])
    tX_normalized = ts_scaler.transform(tX_reshaped).reshape(tX.shape)  # ← TRANSFORM only!
    logger.info(f'Holdout X shape (after normalization): {tX_normalized.shape}')
    
    # 2. TRANSFORM tabular with FITTED scaler
    if num_cols:
        holdout_tab_normalized = holdout.tab_df.copy()
        holdout_tab_normalized[num_cols] = tab_scaler.transform(holdout.tab_df[num_cols])  # ← TRANSFORM only!
    else:
        holdout_tab_normalized = holdout.tab_df

    # ============================================================================
    # HOLDOUT DATALOADERS (with transformed data, no batch transforms)
    # ============================================================================
    logger.info("Creating holdout dataloaders with fixed normalization...")
    
    test_ts_dls = get_ts_dls(
        tX_normalized,  # ← Pre-normalized with trainval scaler!
        ty,
        splits=None,
        tfms=tfms,
        batch_tfms=None,  # ← NO batch transforms!
        bs=cfg["training"]["bs"],
        drop_last=False,
        shuffle=False
    )
    
    logger.info(f'4 {cfg["target"]}')
    
    test_tab_dls = get_tabular_dls(
        holdout_tab_normalized,  # ← Pre-normalized with trainval scaler!
        procs=procs,  # ← Normalize removed!
        cat_names=cat_cols.copy(),
        cont_names=num_cols.copy(),
        y_names=cfg["target"],
        splits=None,
        drop_last=False,
        shuffle=False
    )

    # TRANSFORM holdout using fitted categorical encoder (CRITICAL!)
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
    # RETURN DATA DICT (with scalers for deployment)
    # ============================================================================
    logger.info("Data preparation complete!")
    logger.info(f"  TS scaler: {type(ts_scaler).__name__}")
    logger.info(f"  Tab scaler: {type(tab_scaler).__name__}")
    logger.info(f"  Trainval samples: {len(y)}")
    logger.info(f"  Holdout samples: {len(ty)}")
    
    # ============================================================================
    # VALIDATION: Ensure normalization actually worked
    # ============================================================================
    assert abs(X_normalized.mean()) < 1.0, f"X not normalized! mean={X_normalized.mean():.4f}"
    assert 0.5 < X_normalized.std() < 2.0, f"X std wrong! std={X_normalized.std():.4f}"
    
    if num_cols:
        tab_mean = trainval_tab_normalized[num_cols].mean().mean()
        tab_std = trainval_tab_normalized[num_cols].std().mean()
        assert abs(tab_mean) < 1.0, f"Tabular not normalized! mean={tab_mean:.4f}"
        assert 0.5 < tab_std < 2.0, f"Tabular std wrong! std={tab_std:.4f}"
    
    logger.info("✓ Normalization validation passed!")
    
    return {
        "base": base,
        "trainval": trainval,
        "holdout": holdout,
        "X": X_normalized,  # ← Normalized!
        "X_multi_hot": ts_cat_dls.X_multi_hot, 
        "y": y,
        "tX": tX_normalized,  # ← Normalized!
        "tX_multi_hot": test_ts_cat_dls.X_multi_hot,
        "ty": ty,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "tfms": tfms,
        "batch_tfms": None,  # ← No batch transforms!
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
        
        # ========================================================================
        # NEW: SCALERS FOR DEPLOYMENT AND EVALUATION
        # ========================================================================
        "ts_scaler": ts_scaler,  # ← Fit on trainval, apply to holdout
        "tab_scaler": tab_scaler,  # ← Fit on trainval, apply to holdout
        "ts_feature_names": trainval.complete.columns[3:-1].tolist(),  # For reference
    }


# ============================================================================
# UTILITY: Save scalers for deployment
# ============================================================================

def save_normalization_artifacts(data, model_name, save_dir='models/scalers'):
    """
    Save normalization scalers and metadata for deployment.
    
    Usage:
        save_normalization_artifacts(data, '13012025')
    """
    import os
    import pickle
    
    os.makedirs(save_dir, exist_ok=True)
    
    artifacts = {
        'ts_scaler': data['ts_scaler'],
        'tab_scaler': data['tab_scaler'],
        'ts_feature_names': data.get('ts_feature_names'),
        'tab_feature_names': data['num_cols'],
        'cat_feature_names': data['cat_cols'],
        'encoding_info': data['encoding_info'],
        'cat_encoder': data['cat_encoder'],
        'model_name': model_name,
        'scaler_type': type(data['ts_scaler']).__name__,
    }
    
    save_path = f'{save_dir}/normalization_{model_name}.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(artifacts, f)
    
    logger.info(f"Saved normalization artifacts to {save_path}")
    return save_path


def load_normalization_artifacts(model_name, load_dir='models/scalers'):
    """
    Load saved normalization artifacts.
    
    Usage:
        artifacts = load_normalization_artifacts('13012025')
        ts_normalized = artifacts['ts_scaler'].transform(new_ts_data)
    """
    import pickle
    
    load_path = f'{load_dir}/normalization_{model_name}.pkl'
    with open(load_path, 'rb') as f:
        artifacts = pickle.load(f)
    
    logger.info(f"Loaded normalization artifacts from {load_path}")
    logger.info(f"  Scaler type: {artifacts['scaler_type']}")
    
    return artifacts


# ============================================================================
# UTILITY: Apply normalization to new data (for deployment)
# ============================================================================

def normalize_new_patient(patient_ts_data, patient_tab_data, artifacts):
    """
    Apply saved normalization to new patient data.
    
    Args:
        patient_ts_data: Time series array [seq_len, n_features] or [1, seq_len, n_features]
        patient_tab_data: DataFrame or dict with tabular features
        artifacts: Dict from load_normalization_artifacts()
    
    Returns:
        Normalized (ts_data, tab_data)
    """
    # Handle single patient (add batch dim if needed)
    if patient_ts_data.ndim == 2:
        patient_ts_data = patient_ts_data[np.newaxis, ...]
    
    # Normalize TS
    ts_shape = patient_ts_data.shape
    ts_reshaped = patient_ts_data.reshape(-1, ts_shape[2])
    ts_normalized = artifacts['ts_scaler'].transform(ts_reshaped).reshape(ts_shape)
    
    # Normalize tabular
    num_cols = artifacts['tab_feature_names']
    if num_cols:
        if isinstance(patient_tab_data, dict):
            patient_tab_data = pd.DataFrame([patient_tab_data])
        
        tab_normalized = patient_tab_data.copy()
        tab_normalized[num_cols] = artifacts['tab_scaler'].transform(patient_tab_data[num_cols])
    else:
        tab_normalized = patient_tab_data
    
    return ts_normalized, tab_normalized
