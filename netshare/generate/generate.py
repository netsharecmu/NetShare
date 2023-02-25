import os
from typing import Optional

import pandas as pd

from netshare.generate.denormalize_fields import denormalize_fields
from netshare.generate.generate_api import (
    get_best_generated_data_dir,
    get_generated_data_dir,
    get_raw_generated_data_dir,
)
from netshare.generate.model_generate import model_generate
from netshare.utils.model_configuration import create_chunks_configurations


def generate(target_dir: Optional[str] = None) -> None:
    """
    This is the main function of the postprocess phase.
    We get the generated data, prepare it to be exported, export it, and create the visualization.
    If target_dir is given, then we will export the data to it,
        otherwise, we will export the data to `generate_api.get_best_generated_data_dir()`.

    This function execute the following steps:
    1. Denormalize the fields (e.g. int to IP, vector to word, etc.)
    2. Choose the best generated data and write it using the generate_api
    """
    model_generate()
    denormalize_fields()
    choose_best_model(target_dir)


def choose_best_model(target_dir: Optional[str] = None) -> None:
    """
    :return: the path to the data that was created by the chosen model
        (the raw data that should be shared with the user).
    """
    # Current naive way: for each hyperparameter set, output one synthetic dataset by aggregating last ckpt's data.
    # TODO
    # 1. trimming strategy: trim each chunk or trim the entire csv after aggregation
    # 2. pick best model across hyperparameters/ckpts by metrics

    configs, config_group_list = create_chunks_configurations(generation_flag=True)
    for config_group_idx, config_group in enumerate(config_group_list):
        syn_csvs = []
        for config_idx in config_group["config_ids"]:
            config = configs[config_idx]
            csv_root_folder = config["eval_root_folder"].replace(
                get_raw_generated_data_dir(), get_generated_data_dir()
            )
            csv_folder = os.path.join(
                csv_root_folder,
                f"chunk_id-{config['chunk_id']}",
            )
            max_iteration_id = max(
                [
                    int(os.path.splitext(f)[0].split("-")[1])
                    for f in os.listdir(csv_folder)
                    if f.endswith(".csv")
                ]
            )
            last_iteration_csv = pd.read_csv(
                os.path.join(csv_folder, f"iteration_id-{max_iteration_id}.csv")
            )
            syn_csvs.append(last_iteration_csv)
        # write aggregated csvs (from chunks) to `best_generated` dir
        syn_csv_filename = ",".join(
            [f"{k}-{v}" for k, v in config_group.items() if k not in ["config_ids"]]
            + [
                f"{k}-{v}"
                for k, v in config.items()
                if f"{k}_expand" in config and k not in ["dataset"]
            ]
        )
        syn_csv = pd.concat(syn_csvs)
        # sort by timestamp if exists
        if config["timestamp"].get("generation", False):
            syn_csv.sort_values(by=config["timestamp"]["column"])
        target_dir = target_dir or get_best_generated_data_dir()
        os.makedirs(target_dir, exist_ok=True)
        syn_csv.to_csv(os.path.join(target_dir, syn_csv_filename), index=False)
