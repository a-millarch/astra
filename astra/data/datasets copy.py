from astra.utils import logger, cfg, get_concept
import pandas as pd

class TSDS:
    def __init__(
        self,
        cfg,
        base_df,
        default_mode=True,
        concepts=["VitaleVaerdier", "ITAOversigtsrapport", "Labsvar", "Medicin"],
    ):
        self.cfg = cfg
        self.target = cfg["target"]
        self.base = base_df
        self.concepts = concepts

        if default_mode:
            self.set_tab_df()
            self.collect_concepts()
            

    def set_tab_df(self):
        self.tab_df = self.base[[cfg["dataset"]["id_col"],cfg["target"]]+cfg["dataset"]["num_cols"]+cfg["dataset"]["cat_cols"]]
        self.tab_df[cfg["dataset"]["num_cols"]] = self.tab_df[cfg["dataset"]["num_cols"]].astype(float)
        logger.debug(self.base.columns)


    
    def collect_concepts(self):
        concepts = {}
        concepts_raw = {}
        for concept in self.concepts:
            logger.debug(f"getting {concept}")
            concepts_raw[concept] = get_concept(concept, self.cfg)

            logger.debug(f"getting long version of {concept}")
            if concept in self.cfg["dataset"]["ts_cat_names"]:
                agg_func_name = self.cfg["agg_func"][concept]
                concept_long_df = concepts_raw[concept][cfg["agg_func"][concept][0]].copy(deep=True)
                concepts[concept] = _get_long_concept_df_multi_label(concept_long_df, self.cfg)
            else:
                concepts[concept] = _get_long_concept_df_single_label(
                    self.cfg,
                    self.base.copy(deep=True),
                    concepts_raw[concept],
                    concept,
                    self.cfg["target"],
                    self.cfg["bin_freq_include"],   
                )

        self.concepts = concepts
        self.concepts_raw = concepts_raw

    def change_na_fill(self, mode="forward"): #OBSOLETE?
        if mode == "forward":
            logger.info("Forward filling vitals")
            self.vitals = self.vitals.replace({0.0: np.nan})
            # if first row missing, fill with 0, forward fill the rest
            self.vitals["0"] = self.vitals["0"].fillna(0.0)
            self.vitals.iloc[:, :-1] = self.vitals.iloc[:, :-1].ffill(axis=1)
            # for target and if ffill not available
            self.vitals = self.vitals.fillna(0.0)


def _get_long_concept_df_single_label(
    cfg: Dict,
    base: pd.DataFrame,
    concepts: Dict,
    concept: str,
    target: str,
    bin_freq_include: Optional[List],
    
) -> pd.DataFrame:
    """
    Process single-label data: PIVOT to wide format.
    This is the original behavior.
    """
    print(f"SINGLE LABEL for {concept}")
    pivoted = []
    
    for agg_func in cfg["agg_func"][concept]:
        logger.debug(f"Single-label: {concept} with {agg_func}")
        df = concepts[agg_func].copy()
        
        if bin_freq_include is not None:
            df = df[df.bin_freq.isin(bin_freq_include)]
        
        try:
            df = df[
                (~df.FEATURE.isin(cfg["drop_features"].get(concept, []))) &
                (df.PID.isin(base.PID.unique()))
            ][["PID", "bin_counter", "FEATURE", "VALUE"]]
        except:
            df = df[
                (df.PID.isin(base.PID.unique()))
            ][["PID", "bin_counter", "FEATURE", "VALUE"]]
        
        # Convert to numeric
        try:
            df['VALUE'] = pd.to_numeric(df['VALUE'])
        except (ValueError, TypeError):
            pass  # Keep as string if conversion fails
        
        # Pivot the dataframe (safe because single-label = no duplicates)
        pivoted_df = df.pivot(
            index=["PID", "FEATURE"], 
            columns="bin_counter", 
            values="VALUE"
        )
        
        pivoted_df = pivoted_df.reset_index()
        pivoted_df.columns.name = None
        pivoted_df.columns = ["PID", "FEATURE"] + [
            f"{i}" for i in range(len(pivoted_df.columns) - 2)
        ]
        
        pivoted_df = pivoted_df.sort_values(["PID", "FEATURE"]).reset_index(drop=True)
        
        # Create complete set of PID-FEATURE combinations
        unique_pids = base["PID"].unique()
        unique_features = pivoted_df["FEATURE"].unique()
        
        complete_set = pd.MultiIndex.from_product(
            [unique_pids, unique_features], 
            names=["PID", "FEATURE"]
        )
        complete_df = pd.DataFrame(index=complete_set).reset_index()
        
        merged_df = complete_df.merge(pivoted_df, on=["PID", "FEATURE"], how="left")
        
        # Fill missing values
        numeric_cols = [col for col in merged_df.columns if col.isdigit()]
        if len(numeric_cols) > 0 and pd.api.types.is_numeric_dtype(merged_df[numeric_cols[0]]):
            merged_df[numeric_cols] = merged_df[numeric_cols].fillna(0)
        
        merged_df = merged_df.sort_values(["PID", "FEATURE"])
        
        # Rename features
        col_mapper = {
            feat: f"{feat}_{agg_func}" 
            for feat in df.FEATURE.unique()
        }
        merged_df["FEATURE"] = merged_df["FEATURE"].replace(col_mapper)
        
        pivoted_df = merged_df.reset_index(drop=True)
        pivoted_df = pivoted_df[pivoted_df.FEATURE.notnull()]
        
        pivoted.append(pivoted_df)
    
    # Concat all aggregation functions
    complete = pd.concat(pivoted, ignore_index=True)
    
    # Fill NaN in numeric columns
    numeric_cols = [col for col in complete.columns if col.isdigit()]
    for col in numeric_cols:
        if pd.api.types.is_numeric_dtype(complete[col]):
            complete[col] = complete[col].fillna(0.0)
    
    # Merge target
    prelen = len(complete)
    complete = complete.merge(base[["PID", target]].copy(deep=True), on="PID", how="left")
    complete[target] = complete[target].astype(int)
    assert prelen == len(complete), f"Length mismatch: {prelen} vs {len(complete)}"
    
    complete = complete[complete.FEATURE.notnull()].reset_index(drop=True)
    
    return complete


def _get_long_concept_df_multi_label(df_long:pd.DataFrame, cfg):
    df_long = df_long[df_long.bin_freq.isin(cfg["bin_freq_include"])].copy(deep=True)
    df_long = df_long[['PID', 'bin_counter','FEATURE', 'VALUE']].rename(columns={'bin_counter':'TIMESTEP'})
    df_long["TIMESTEP"] = df_long["TIMESTEP"]-1 # matching df2xy function index 0
    # Pivot to wide format (your format)
    df_wide = df_long.pivot_table(
      index=['PID', 'FEATURE'],
      columns='TIMESTEP',
      values='VALUE',
      aggfunc=lambda x: list(x) if len(x) > 1 else x.iloc[0],
      dropna=False
    ).reset_index()
    # keep all timesteps, fill in feature name
    feat_name = df_wide.FEATURE.dropna().unique()
    assert len(feat_name == 1)
    df_wide['FEATURE'] = feat_name[0]
    df_wide.timestep_cols = [i for i in range(0,df_long.TIMESTEP.max())]
    return df_wide






class HistoricDS:
    def __init__(self, cfg, base_df):
        self.cfg = cfg
        self.base = base_df
        self.target = cfg["target"]
        self.tab_df = self.base[[cfg["dataset"]["id_col"],cfg["target"]]+cfg["dataset"]["num_cols"]+cfg["dataset"]["cat_cols"]]
        self.tab_df[cfg["dataset"]["num_cols"]] = self.tab_df[cfg["dataset"]["num_cols"]].astype(float)
        logger.debug(self.base.columns)
    
    def add_comorbidity(self):
        """Add Elixhauser and ISS to base"""
        
        # ELIXHAUSER    
        while True:
            try:
                elix = pd.read_csv('data/interim/ISS_ELIX/computed_elix_df.csv',
                                   low_memory =False)
                logger.info('Elixhauser df dataframe found, continuing') 
                baselen =len(self.base)
                # merge
                self.base= self.base.merge(elix[["PID", "elixscore"]], 
                                           how='left', on='PID')
                assert baselen-len(self.base) == 0
                logger.info('Merged Elix onto base')
            # TODO: merge onto base
            except FileNotFoundError:
                logger.info('No Elixhauser computed, creating.')
                add_elixhauser(self.base)
                continue
            break
           


def get_long_concept_df(
    base: pd.DataFrame,
    concepts: dict,
    concept: str,
    target: str,
    bin_freq_include: list = None,) -> pd.DataFrame:
    pivoted = []
    for agg_func in cfg["agg_func"][concept]:
        logger.debug(f"long df {agg_func}")
        df = concepts[agg_func]
        if bin_freq_include is not None:
            logger.debug("Reducing to limited bin frequencies")
            df = df[df.bin_freq.isin(bin_freq_include)]
            #logger.debug(print(df))
        try:
            df = df[
                (~df.FEATURE.isin(cfg["drop_features"][concept]))
                & (df.PID.isin(base.PID.unique()))
            ][["PID", "bin_counter", "FEATURE", "VALUE"]]
        except:
            df = df[(df.PID.isin(base.PID.unique()))][
                ["PID", "bin_counter", "FEATURE", "VALUE"]
            ]



        # Pivot the dataframe
        pivoted_df = df.pivot(
            index=["PID", "FEATURE"], columns="bin_counter", values="VALUE"
        )

        # Reset the index to make PID and FEATURE regular columns
        pivoted_df = pivoted_df.reset_index()

        # Rename the columns (bin_counter columns will be 0, 1, 2, ...)
        pivoted_df.columns.name = None
        pivoted_df.columns = ["PID", "FEATURE"] + [
            f"{i}" for i in range(len(pivoted_df.columns) - 2)
        ]

        # Sort the dataframe by PID and FEATURE
        pivoted_df = pivoted_df.sort_values(["PID", "FEATURE"])

        # Reset the index to have a clean, sequential index
        pivoted_df = pivoted_df.reset_index(drop=True)

        # Step 1: Get unique PIDs and FEATUREs
        # important: use an absolute lit of PIDS e.g. by base
        unique_pids = base["PID"].unique()
        unique_features = pivoted_df["FEATURE"].unique()

        # Step 2: Create a complete set of PID-FEATURE combinations
        complete_set = pd.MultiIndex.from_product(
            [unique_pids, unique_features], names=["PID", "FEATURE"]
        )
        complete_df = pd.DataFrame(index=complete_set).reset_index()

        # Step 3: Merge the complete set with the original dataframe
        merged_df = complete_df.merge(pivoted_df, on=["PID", "FEATURE"], how="left")

        # Step 4: Fill missing values in numeric columns with 0
        numeric_cols = [col for col in merged_df.columns if col.isdigit()]
        merged_df[numeric_cols] = merged_df[numeric_cols].fillna(0)

        # Sort the dataframe if needed
        merged_df = merged_df.sort_values(["PID", "FEATURE"])

        # Rename feature values for later concat
        col_mapper = dict(
            zip(
                df.FEATURE.unique(),
                [col + f"_{agg_func}" for col in df.FEATURE.unique()],
            )
        )
        merged_df["FEATURE"] = merged_df["FEATURE"].replace(col_mapper)
        # Reset the index
        pivoted_df = merged_df.reset_index(drop=True)

        pivoted_df = pivoted_df[pivoted_df.FEATURE.notnull()]

        pivoted.append(pivoted_df)
    # Concat all features into on df
    complete = pd.concat(pivoted).fillna(0.0)

    # Merge target onto long df
    prelen = len(complete)
    complete = complete.merge(base[["PID", target]], on="PID", how="left")
    complete[target] = complete[target].astype(int)
    assert prelen - len(complete) == 0

    # Quality control and reset index
    complete = complete[complete.FEATURE.notnull()]
    complete = complete.reset_index(drop=True)

    return complete