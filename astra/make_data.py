import pandas as pd
import numpy as np
from cstar import ProjectManager
import os 

from astra.utils import cfg
from astra.utils import is_file_present, are_files_present

from astra.data.collectors import collect_subsets
import astra.data.build_patient_info as bpi
from astra.data.filters import filter_subsets_inhospital
from astra.data.mapper import map_concept

from astra.data.datasets import TSDS

def generate_base_df():
    # JUST A TEMPORARY TESTER FUNCTION, used by load_or_collect_population
    
    pd.DataFrame.from_dict({'CPR_hash':['CENSORED'],
    'ServiceDate':[np.datetime64('2026-0120T00:00:00.000000000')]}, orient='columns').to_csv('data/external/trauma_call.csv')
    
   
    # saved as pickle

def proces_raw_concepts(cfg, base= None, reset=False): # move to construct data_sets?
    subsets_filenames = cfg["default_load_filenames"] + cfg["large_load_filenames"]
    if (
            are_files_present("data/raw", subsets_filenames, extension=".csv")
            and reset == False
        ):
            logger.info("All subsets found, continuing")
    else:
            logger.info("Subsets missing, collecting missing")
            collect_subsets(cfg, base=base)

def proces_inhospital_concepts(cfg, reset=False):
    subsets_filenames = cfg["default_load_filenames"] + cfg["large_load_filenames"]
    if (
        are_files_present("data/interim/concepts", subsets_filenames, extension=".pkl")
        and reset == False
    ):
        logger.info("Interim subsets found, continuing")
    else:
        logger.info("Filtering subsets")
        filter_subsets_inhospital(cfg)
    
def map_data(cfg):
    logger.info("Mapping data to bins")
    map_dir = "data/interim/mapped/"
    for concept in cfg["concepts"]:
        for agg_func in cfg["agg_func"][concept]:
            if is_file_present(
                f"{map_dir}{concept}_{agg_func}.csv"
            ) and is_file_present(f"{map_dir}{concept}_{agg_func}.pkl"):
                pass
            else:
                logger.debug(f"Binning and mapping {concept} with agg_func: {agg_func}")
                map_concept(cfg, concept, agg_func)

            
if __name__ =='__main__':
    pm = ProjectManager()
    logger = pm.setup_logging(print_only=True)
    #generate_base_df() #Simulates new patient drop
  

    population = bpi.load_or_collect_population(cfg)
    proces_raw_concepts(cfg, base=population)
    
    base = bpi.create_base_df(cfg)
    bpi.create_bin_df(cfg)

    proces_inhospital_concepts(cfg, reset=False)
    map_data(cfg)
    logger.info("Creating TSDS")
    tsds = TSDS(cfg, base)
    logger.info(tsds.concepts.keys())
    
    
   