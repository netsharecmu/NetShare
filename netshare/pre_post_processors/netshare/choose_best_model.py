import os
import json

import pandas as pd


def choose_best_model(
    config_pre_post_processor,
    pre_processed_data_folder,
    generated_data_folder,
    post_processed_data_folder
):
    with open(os.path.join(generated_data_folder, "configs_generate.json"), 'r') as f:
        data = json.load(f)
        configs = data["configs"]
        config_group_list = data["config_group_list"]

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
