import os
import json

import pandas as pd
import numpy as np

from .util import create_sdmetrics_config
from sdmetrics.reports.timeseries import QualityReport


def compare_rawdf_syndfs(
    raw_df,
    syn_dfs,
    config_pre_post_processor
):
    sdmetrics_config = create_sdmetrics_config(
        config_pre_post_processor,
        comparison_type='quantitative')
    report = QualityReport(config_dict=sdmetrics_config['config'])
    report.generate(raw_df, syn_dfs[0], sdmetrics_config['metadata'])
    print("\n\n\n", report.dict_metric_scores)

    ()+1


def choose_best_model(
    config_pre_post_processor,
    pre_processed_data_folder,
    generated_data_folder,
    post_processed_data_folder
):
    with open(os.path.join(generated_data_folder, "configs_generate.json"), 'r') as f:
        data = json.load(f)
        configs = data["configs"]
        # add pre_processor configs in place
        for config in configs:
            config.update(config_pre_post_processor)
        config_group_list = data["config_group_list"]

    # TODO: change to distribute (Ray-style)
    dict_dataset_syndfs = {}
    for config_group_idx, config_group in enumerate(config_group_list):
        print("Config group #{}: {}".format(config_group_idx, config_group))
        config_ids = config_group["config_ids"]
        chunk0_idx = config_ids[0]
        syndf_root_folder = os.path.join(
            configs[chunk0_idx]["eval_root_folder"], "syn_dfs"
        )
        assert len(
            [
                file
                for file in os.listdir(syndf_root_folder)
                if file.startswith("chunk_id")
            ]
        ) == len(config_ids)

        best_syndfs = []
        truncate_ratios = []
        for chunk_id, config_idx in enumerate(config_ids):
            config = configs[config_idx]
            raw_df = pd.read_csv(os.path.join(config["dataset"], "raw.csv"))
            time_col_name = getattr(
                getattr(config_pre_post_processor, 'timestamp'),
                'column')

            syn_dfs = []
            syn_dfs_names = []
            syn_df_folder = os.path.join(
                syndf_root_folder, "chunk_id-{}".format(chunk_id)
            )
            for file in os.listdir(syn_df_folder):
                if file.endswith(".csv"):
                    syn_dfs_names.append(file)
                    syn_df = pd.read_csv(os.path.join(syn_df_folder, file))

                    # truncate to raw data time range
                    if config["truncate"] == "per_chunk":
                        syn_df_truncated = syn_df[
                            (syn_df[time_col_name] >= raw_df[time_col_name].min())
                            & (syn_df[time_col_name] <= raw_df[time_col_name].max())
                        ]
                    # TODO: support more truncation methods if necessary
                    else:
                        raise ValueError("Unknown truncation methods...")
                    truncate_ratios.append(
                        1.0 - len(syn_df_truncated)/len(syn_df))

                    syn_dfs.append(syn_df_truncated)

        print("Average truncation ratio:", np.mean(truncate_ratios))
        compare_rawdf_syndfs(
            raw_df[syn_dfs[0].columns], syn_dfs, config_pre_post_processor
        )
