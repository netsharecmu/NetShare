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
