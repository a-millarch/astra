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
            concepts[concept] = get_long_concept_df(
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


def get_long_concept_df(
    base: pd.DataFrame,
    vitals: dict,
    concept: str,
    target: str,
    bin_freq_include: list = None,) -> pd.DataFrame:
    pivoted = []
    for agg_func in cfg["agg_func"][concept]:
        logger.debug(f"long df {agg_func}")
        df = vitals[agg_func]
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