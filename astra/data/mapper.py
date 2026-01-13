import pandas as pd
import numpy as np

from typing import List, Dict, Optional

from astra.utils import logger

def merge_and_aggregate(
    bin_df: pd.DataFrame,
    subset_df: pd.DataFrame,
    agg_func: str = "mean",
    is_categorical: bool = False,
    is_multi_label: bool = False
) -> pd.DataFrame:
    """
    Enhanced merge and aggregate that handles both continuous and categorical data.
    
    Args:
        bin_df: DataFrame with time bins
        subset_df: DataFrame with values to aggregate
        agg_func: Aggregation function for continuous data
        is_categorical: Whether this feature is categorical
        is_multi_label: Whether this categorical feature can have multiple values per bin
    
    Returns:
        Aggregated DataFrame
    """
    # Ensure datatypes for memory efficiency
    bin_df["PID"] = bin_df["PID"].astype("int32")
    subset_df["PID"] = subset_df["PID"].astype("int32")
    
    # For continuous: ensure numeric
    # For categorical: keep as object/string
    if not is_categorical:
        subset_df["VALUE"] = subset_df["VALUE"].astype("float")
    
    # Merge on PID
    merged_df = pd.merge(bin_df, subset_df, on="PID", how="left")
    
    # Filter based on timestamp conditions
    filtered_df = merged_df[
        (merged_df["TIMESTAMP"] >= merged_df["bin_start"])
        & (merged_df["TIMESTAMP"] <= merged_df["bin_end"])
    ]
    
    # === CATEGORICAL HANDLING (NEW) ===
    if is_categorical:
        if is_multi_label:
            # Keep all values as separate rows
            aggregated_df = filtered_df[
                ["PID", "bin_counter", "bin_start", "bin_end", "FEATURE", "VALUE"]
            ].drop_duplicates()

        else:
            # Single-label: Take mode (most common) or last value
            logger.debug(f"Single-label categorical: using mode/last")
            
            if agg_func == "mode":
                aggregated_df = (
                    filtered_df.groupby(["PID", "bin_counter", "bin_start", "bin_end", "FEATURE"])
                    .agg({"VALUE": lambda x: x.mode()[0] if len(x.mode()) > 0 else np.nan})
                    .reset_index()
                )
            elif agg_func == "last":
                aggregated_df = (
                    filtered_df.groupby(["PID", "bin_counter", "bin_start", "bin_end", "FEATURE"])
                    .agg({"VALUE": "last"})
                    .reset_index()
                )
            elif agg_func == "first":
                aggregated_df = (
                    filtered_df.groupby(["PID", "bin_counter", "bin_start", "bin_end", "FEATURE"])
                    .agg({"VALUE": "first"})
                    .reset_index()
                )
            else:
                # Default to last
                logger.warning(f"Unknown agg_func '{agg_func}' for categorical, using 'last'")
                aggregated_df = (
                    filtered_df.groupby(["PID", "bin_counter", "bin_start", "bin_end", "FEATURE"])
                    .agg({"VALUE": "last"})
                    .reset_index()
                )
    
    # === CONTINUOUS HANDLING (ORIGINAL) ===
    else:
        aggregation = {
            "first": "first",
            "mean": "mean",
            "max": "max",
            "min": "min",
            "std": "std",
            "sum": "sum",
            "count": "count",
            "last": "last",
        }
        agg_function = aggregation.get(agg_func, "mean")
        
        aggregated_df = (
            filtered_df.groupby(["PID", "bin_counter", "bin_start", "bin_end", "FEATURE"])
            .agg({"VALUE": agg_function})
            .reset_index()
        )
    
    # Merge the result back to bin_df to maintain all rows
    result_df = pd.merge(
        bin_df,
        aggregated_df,
        on=["PID", "bin_counter", "bin_start", "bin_end"],
        how="left",
    )
    
    return result_df


def map_concept(
    cfg: Dict,
    concept: str,
    agg_func: str,
    is_categorical: bool = False,
    is_multi_label: bool = False
) -> None:
    """
    Enhanced map_concept that handles categorical features.
    
    Args:
        cfg: Configuration dictionary
        concept: Name of concept (e.g., 'Medicin', 'VitaleVaerdier')
        agg_func: Aggregation function
        is_categorical: Whether this concept contains categorical features
        is_multi_label: Whether categorical features can have multiple values per bin
    """
    from astra.utils import get_bin_df
    from astra.data.filters import collect_filter
    
    output_path = f"data/interim/mapped/{concept}"
    
    # Load binning DataFrame
    bin_df = get_bin_df()
    logger.info(f"Prepared bin df for {concept} (categorical={is_categorical}, multi_label={is_multi_label})")
    
    # Load and filter concept
    concept_df = pd.read_pickle(f"data/interim/concepts/{concept}.pkl")
    filter_function = collect_filter(concept)
    concept_df = filter_function(concept_df)
    
    # Process each feature
    dfs = []
    logger.info(f"Processing {len(concept_df.FEATURE.unique())} features")
    
    for feat in concept_df.FEATURE.unique():
        logger.info(f"Processing feature: {feat}")
        subset = concept_df[concept_df.FEATURE == feat]
        
        # Merge and aggregate with categorical handling
        result_df = merge_and_aggregate(
            bin_df, 
            subset, 
            agg_func=agg_func,
            is_categorical=is_categorical,
            is_multi_label=is_multi_label
        )
        
        dfs.append(result_df)
    
    logger.info("Concatenating feature dataframes")
    
    # Concatenate and save
    if len(dfs) < 1:
        logger.warning(f"Concept {concept} failed - no features processed")
        bin_df["FEATURE"] = np.nan
        bin_df["VALUE"] = np.nan
        bin_df.to_pickle(f"{output_path}_{agg_func}.pkl")
        bin_df.to_csv(f"{output_path}_{agg_func}.csv", index=False)
    else:
        result_df = (
            pd.concat(dfs)
            .drop_duplicates()
            .sort_values(["PID", "bin_counter"])
            .reset_index(drop=True)
        )
        
        # === HANDLE MULTI-LABEL EXPANSION (NEW) ===
        if is_categorical and is_multi_label:
            logger.info("Expanding multi-label values")
            
            # Expand lists into separate rows
            # This creates multiple rows per PID/bin/feature for multi-label
            expanded_rows = []
            
            for idx, row in result_df.iterrows():
                if pd.notna(row['VALUE']):
                    # Check if VALUE is a list
                    if isinstance(row['VALUE'], list):
                        # Create separate row for each value in list
                        for val in row['VALUE']:
                            new_row = row.copy()
                            new_row['VALUE'] = val
                            expanded_rows.append(new_row)
                    else:
                        # Single value, keep as is
                        expanded_rows.append(row)
                else:
                    # NaN value, keep as is
                    expanded_rows.append(row)
            
            result_df = pd.DataFrame(expanded_rows).reset_index(drop=True)
            logger.info(f"Expanded to {len(result_df)} rows (from multi-label)")
        
        # Remove bin placeholder/nan rows if data rows present
        grouped = result_df.groupby(["PID", "bin_counter"])
        
        def filter_rows(group):
            """Keep NaN rows only if no data exists for that group"""
            if group["FEATURE"].isna().all() and group["VALUE"].isna().all():
                return group
            else:
                return group.dropna(subset=["FEATURE", "VALUE"])
        
        logger.info(f"Cleaning binned {concept} dataframe")
        filtered_df = grouped.apply(filter_rows).reset_index(drop=True)
        
        # Log statistics
        logger.info(f"Final shape: {filtered_df.shape}")
        if is_categorical:
            n_unique_values = filtered_df['VALUE'].nunique()
            logger.info(f"Unique categorical values: {n_unique_values}")
            if is_multi_label:
                avg_values_per_bin = filtered_df.groupby(['PID', 'bin_counter']).size().mean()
                logger.info(f"Average values per bin: {avg_values_per_bin:.2f}")
        
        logger.info(f"Saving file to {output_path}")
        filtered_df.to_pickle(f"{output_path}_{agg_func}.pkl")
        filtered_df.to_csv(f"{output_path}_{agg_func}.csv", index=False)


def map_all_concepts(cfg: Dict, force: bool = False) -> None:
    """
    Map all concepts defined in config with proper categorical handling.
    
    Args:
        cfg: Configuration dictionary with:
            - concepts: List of concept names
            - agg_func: Dict mapping concept to list of agg functions
            - cat_time_series: Dict with categorical configuration
        force: Whether to reprocess existing files
    """
    import os
    from pathlib import Path
    
    # Get categorical configuration
    cat_config = cfg.get("cat_time_series", {})
    cat_concepts = cat_config.get("concepts", {})
    multi_label_concepts = cat_config.get("multi_label", [])
    
    logger.info("="*80)
    logger.info("MAPPING ALL CONCEPTS")
    logger.info("="*80)
    
    concepts = cfg.get("concepts", [])
    
    for concept in concepts:
        # Determine if concept is categorical
        is_categorical = concept in cat_concepts
        is_multi_label = concept in multi_label_concepts
        
        if is_categorical:
            logger.info(f"\n{concept}: CATEGORICAL" + 
                       (" + MULTI-LABEL" if is_multi_label else ""))
        else:
            logger.info(f"\n{concept}: CONTINUOUS")
        
        # Get aggregation functions for this concept
        agg_funcs = cfg["agg_func"].get(concept, ["mean"])
        
        for agg_func in agg_funcs:
            output_file = f"data/interim/mapped/{concept}_{agg_func}.csv"
            
            # Check if file exists
            if not force and os.path.exists(output_file):
                logger.info(f"  {agg_func}: Already exists, skipping")
                continue
            
            logger.info(f"  Processing with agg_func: {agg_func}")
            
            try:
                map_concept(
                    cfg=cfg,
                    concept=concept,
                    agg_func=agg_func,
                    is_categorical=is_categorical,
                    is_multi_label=is_multi_label
                )
                logger.info(f"  ✓ {concept}_{agg_func} completed")
            except Exception as e:
                logger.error(f"  ✗ {concept}_{agg_func} failed: {str(e)}")
                raise
    
    logger.info("\n" + "="*80)
    logger.info("ALL CONCEPTS MAPPED SUCCESSFULLY")
    logger.info("="*80)


# ============================================================================
# Utility Functions
# ============================================================================

def check_concept_type(concept_path: str) -> Dict[str, bool]:
    """
    Analyze a concept to determine if it's categorical.
    
    Args:
        concept_path: Path to concept pickle file
    
    Returns:
        Dict with analysis results
    """
    df = pd.read_pickle(concept_path)
    
    results = {}
    
    for feature in df['FEATURE'].unique():
        feature_data = df[df['FEATURE'] == feature]['VALUE']
        
        # Try to convert to numeric
        numeric_data = pd.to_numeric(feature_data, errors='coerce')
        
        # Calculate metrics
        n_unique = feature_data.nunique()
        n_total = len(feature_data)
        unique_ratio = n_unique / n_total if n_total > 0 else 0
        
        # Check if conversion failed (indicates categorical)
        n_non_numeric = numeric_data.isna().sum() - feature_data.isna().sum()
        is_string = feature_data.dtype == 'object'
        
        # Heuristic: categorical if:
        # 1. Contains non-numeric values, OR
        # 2. Low cardinality (< 20% unique values) and is object type
        is_categorical = (n_non_numeric > 0) or (is_string and unique_ratio < 0.2 and n_unique < 100)
        
        results[feature] = {
            'is_categorical': is_categorical,
            'n_unique': n_unique,
            'unique_ratio': unique_ratio,
            'n_non_numeric': n_non_numeric,
            'dtype': str(feature_data.dtype)
        }
    
    return results


def suggest_categorical_concepts(cfg: Dict) -> Dict[str, List[str]]:
    """
    Analyze all concepts and suggest which should be categorical.
    
    Args:
        cfg: Configuration dictionary
    
    Returns:
        Dict mapping concept names to list of categorical features
    """
    concepts = cfg.get("concepts", [])
    suggestions = {}
    
    logger.info("Analyzing concepts for categorical features...")
    logger.info("="*80)
    
    for concept in concepts:
        concept_path = f"data/interim/concepts/{concept}.pkl"
        
        try:
            results = check_concept_type(concept_path)
            
            categorical_features = [
                feat for feat, info in results.items() 
                if info['is_categorical']
            ]
            
            if categorical_features:
                suggestions[concept] = categorical_features
                
                logger.info(f"\n{concept}: Found {len(categorical_features)} categorical features")
                for feat in categorical_features[:5]:  # Show first 5
                    info = results[feat]
                    logger.info(f"  - {feat}: {info['n_unique']} unique values, "
                              f"{info['n_non_numeric']} non-numeric")
                
                if len(categorical_features) > 5:
                    logger.info(f"  ... and {len(categorical_features) - 5} more")
        
        except Exception as e:
            logger.warning(f"Could not analyze {concept}: {str(e)}")
    
    logger.info("\n" + "="*80)
    logger.info("SUGGESTIONS:")
    logger.info("Add to your config:")
    logger.info("cfg['cat_time_series'] = {")
    logger.info("    'concepts': {")
    
    for concept, features in suggestions.items():
        if len(features) == len(results):
            logger.info(f"        '{concept}': ['all'],  # All features categorical")
        else:
            logger.info(f"        '{concept}': {features},")
    
    logger.info("    },")
    logger.info("    'multi_label': [],  # Add concepts that can have multiple values per bin")
    logger.info("}")
    logger.info("="*80)
    
    return suggestions
