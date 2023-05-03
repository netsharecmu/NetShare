import os
import json

import pandas as pd
import numpy as np

from scipy.stats import rankdata
from .util import create_sdmetrics_config, convert_sdmetricsConfigQuant_to_fieldValueDict
from sdmetrics.reports.timeseries import QualityReport


def compare_rawdf_syndfs(
    raw_df,
    syn_dfs,
    config_pre_post_processor
):
    # Compare raw_df and syn_dfs and return the best syn_df
    sdmetrics_config = create_sdmetrics_config(
        config_pre_post_processor,
        comparison_type='quantitative')
    report = QualityReport(config_dict=sdmetrics_config['config'])

    metrics_dict_list = []
    for syn_df in syn_dfs:
        report.generate(
            raw_df, syn_df, sdmetrics_config['metadata'])
        metrics_dict_list.append(
            convert_sdmetricsConfigQuant_to_fieldValueDict(
                report.dict_metric_scores))

    metrics = list(metrics_dict_list[0].keys())
    metric_vals_dict = {}
    for metrics_dict in metrics_dict_list:
        for metric in metrics:
            if metric not in metric_vals_dict:
                metric_vals_dict[metric] = []
            metric_vals_dict[metric].append(metrics_dict[metric])
    metric_vals_2d = []
    for metric, vals in metric_vals_dict.items():
        metric_vals_2d.append(vals)
    rankings_sum = np.sum(rankdata(metric_vals_2d, axis=1), axis=0)
    best_syndf_idx = np.argmin(rankdata(rankings_sum))

    return best_syndf_idx, syn_dfs[best_syndf_idx]


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
                        1.0 - len(syn_df_truncated) / len(syn_df))

                    syn_dfs.append(syn_df_truncated)

            best_syndf_idx, best_syndf = compare_rawdf_syndfs(
                raw_df[syn_dfs[0].columns], syn_dfs, config_pre_post_processor
            )

            best_syndfs.append(best_syndf)
            print(
                "Chunk_id: {}, # of syn dfs: {}, best_syndf: {}".format(
                    chunk_id, len(syn_dfs), syn_dfs_names[best_syndf_idx]
                )
            )

        print("Average truncation ratio:", np.mean(truncate_ratios))
        big_best_syndf = pd.concat(best_syndfs)
        print("Big syndf shape:", big_best_syndf.shape)
        print()

        if config_group["dp_noise_multiplier"] not in dict_dataset_syndfs:
            dict_dataset_syndfs[config_group["dp_noise_multiplier"]] = []
        dict_dataset_syndfs[config_group["dp_noise_multiplier"]].append(
            big_best_syndf)

        dict_dataset_bestsyndf = {}

    big_raw_df = pd.read_csv(os.path.join(pre_processed_data_folder, "raw.csv"))
    for dpnoisemultiplier, syn_dfs in dict_dataset_syndfs.items():
        assert len(syn_dfs) >= 1
        if len(syn_dfs) > 1:
            best_syndf_idx, best_syn_df = compare_rawdf_syndfs(
                big_raw_df[syn_dfs[0].columns],
                syn_dfs, config_pre_post_processor)
            dict_dataset_bestsyndf[dpnoisemultiplier] = best_syn_df
        else:
            dict_dataset_bestsyndf[dpnoisemultiplier] = syn_dfs[0]

    print("Aggregated final dataset syndf")
    for dp_noise_multiplier, best_syndf in dict_dataset_bestsyndf.items():
        print(dp_noise_multiplier, best_syndf.shape)
        best_syndf_folder = post_processed_data_folder
        os.makedirs(best_syndf_folder, exist_ok=True)

        # find best syndf index i.e., for evaluation fairness
        cur_max_idx = None
        for file in os.listdir(best_syndf_folder):
            if file.startswith(
                "syn_df,dp_noise_multiplier-{},truncate-{},id-".format(
                    dp_noise_multiplier, config["truncate"]
                )
            ):
                this_id = int(os.path.splitext(file)[
                              0].split(",")[-1].split("-")[1])
                if cur_max_idx is None or this_id > cur_max_idx:
                    cur_max_idx = this_id
        if cur_max_idx is None:
            cur_max_idx = 1
        else:
            cur_max_idx += 1

        best_syndf_filename = os.path.join(
            best_syndf_folder,
            "syn_df,dp_noise_multiplier-{},truncate-{},id-{}.csv".format(
                dp_noise_multiplier, config["truncate"], cur_max_idx)
        )
        # best_syndf_filename = os.path.join(best_syndf_folder, "syn.csv")

        print("best_syn_df filename:", best_syndf_filename)

        # sort by timestamp if applicable
        if config_pre_post_processor.timestamp.generation:
            time_col_name = config_pre_post_processor.timestamp.column
        best_syndf = best_syndf.sort_values(time_col_name)
        best_syndf.to_csv(best_syndf_filename, index=False)
