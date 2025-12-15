import pandas as pd
import numpy as np

from astra.utils import get_bin_df, cfg, logger
from astra.data.filters import collect_filter

def merge_and_aggregate(bin_df, subset_df, agg_func="mean"):
    # Ensure datatypes for memory efficiency
    bin_df["PID"] = bin_df["PID"].astype("int32")
    subset_df["PID"] = subset_df["PID"].astype("int32")
    subset_df["VALUE"] = subset_df["VALUE"].astype("float")

    # Merge on PID
    merged_df = pd.merge(bin_df, subset_df, on="PID", how="left")

    # Filter based on timestamp conditions
    filtered_df = merged_df[
        (merged_df["TIMESTAMP"] >= merged_df["bin_start"])
        & (merged_df["TIMESTAMP"] <= merged_df["bin_end"])
    ]

    # Aggregate the values
    aggregation = {
        "first": "first",
        "mean": "mean",
        "max": "max",
        "min": "min",
        "std": "std",
        "sum": "sum",
        "count": "count",
    }
    agg_function = aggregation.get(agg_func, "_")

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

def map_concept(cfg, concept: str, agg_func):
    output_path = f"data/interim/mapped/{concept}"
    
    # Load binning DataFrame
    bin_df = get_bin_df()
    logger.info(f"Prepared bin df, now preparing {concept}")

    # Load and filter concept
    concept_df = pd.read_pickle(f"data/interim/concepts/{concept}.pkl")
    filter_function = collect_filter(concept)
    concept_df = filter_function(concept_df)

    # Process each feature
    dfs = []
    logger.info("Processing each feature")

    for feat in concept_df.FEATURE.unique():
        logger.info(f"start {feat}")
        subset = concept_df[concept_df.FEATURE == feat]

        # Merge and aggregate
        logger.info(f"{feat} merging and aggregating")
        result_df = merge_and_aggregate(bin_df, subset, agg_func=agg_func)
        
        dfs.append(result_df)

    logger.info("Concatenating feature dfs")
    # concat dataframes into one long
    if len(dfs)<1:
        logger.warning(f"concept {concept} failed.")
        bin_df["FEATURE"] = np.nan
        bin_df["VALUE"]=np.nan
        bin_df.to_pickle(f"{output_path}_{agg_func}.pkl")
        bin_df.to_csv(f"{output_path}_{agg_func}.csv", index=False)

    else:
        result_df = (
            pd.concat(dfs)
            .drop_duplicates()
            .sort_values(["PID", "bin_counter"])
            .reset_index(drop=True)
        )
    
        # remove bin placeholder/nan rows if data rows present
        grouped = result_df.groupby(["PID", "bin_counter"])
    
        # Function to filter rows
        def filter_rows(group):
            if group["FEATURE"].isna().all() and group["VALUE"].isna().all():
                return group
            else:
                return group.dropna(subset=["FEATURE", "VALUE"])
    
        logger.info(f"Cleaning binned {concept} dataframe")
        # Applying the filter function to each group
        filtered_df = grouped.apply(filter_rows).reset_index(drop=True)
        
        logger.info(f"Saving file to {output_path}")
        filtered_df.to_pickle(f"{output_path}_{agg_func}.pkl")
        filtered_df.to_csv(f"{output_path}_{agg_func}.csv", index=False)
