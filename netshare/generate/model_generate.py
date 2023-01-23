import copy
import os
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

import netshare.utils.ray as ray
from netshare import models
from netshare.configs import get_config
from netshare.learn import learn_api
from netshare.utils.logger import TqdmToLogger, logger
from netshare.utils.model_configuration import create_chunks_configurations
from netshare.utils.paths import get_generated_data_log_folder


def model_generate() -> None:
    configs, config_group_list = create_chunks_configurations(generation_flag=True)

    logger.info("Start generating attributes")
    ray.get(
        [
            generate_data.remote(
                config=config,
                given_data_attribute_flag=False,
            )
            for config in configs
        ]
    )

    logger.info("Start merging attributes")
    ray.get(
        [
            merge_attr.remote(config_group=config_group, configs=configs)
            for config_group in config_group_list
        ]
    )

    logger.info("Start generating features given attributes")
    ray.get(
        [
            generate_data.remote(
                config=config,
                given_data_attribute_flag=True,
            )
            for config in configs
        ]
    )


@ray.remote(scheduling_strategy="SPREAD", max_calls=1)
def generate_data(
    config: dict,
    given_data_attribute_flag: bool,
) -> None:
    config["given_data_attribute_flag"] = given_data_attribute_flag
    model = models.build_model_from_config()(config)
    model.generate(
        input_train_data_folder=config["dataset"],
        input_model_folder=config["result_folder"],
        output_syn_data_folder=config["eval_root_folder"],
        log_folder=get_generated_data_log_folder(),
    )


@ray.remote(scheduling_strategy="SPREAD", max_calls=1)
def merge_attr(config_group: dict, configs: List[dict]) -> None:
    """
    This is the function to merge the attributes. The models trained on different
    chunks of data will first generate attributes. Then, this function will be
    invoked to merge the synthetic attributes. Given the merged attributes, models
    will continue generating the time-series measurements.

    :return: None
    """
    chunk0_idx = config_group["config_ids"][0]
    eval_root_folder = configs[chunk0_idx]["eval_root_folder"]
    attr_raw_npz_folder = os.path.join(eval_root_folder, "attr_raw")
    num_chunks = len(config_group["config_ids"])

    dim = 0
    for field in list(learn_api.get_attributes_fields().values()):
        if field.name != "startFromThisChunk":
            dim += field.get_output_dim()
        else:
            break
    col_idx_of_session_start = dim

    attr_clean_npz_folder = os.path.join(
        str(Path(attr_raw_npz_folder).parents[0]), "attr_clean"
    )
    os.makedirs(attr_clean_npz_folder, exist_ok=True)

    dict_chunkid_attr: Dict[int, List[List[float]]] = {}
    for chunkid in range(num_chunks):
        dict_chunkid_attr[chunkid] = []

    for chunkid in tqdm(range(num_chunks), file=TqdmToLogger("merge_attr")):
        n_flows_startFromThisEpoch = 0

        if not os.path.exists(
            os.path.join(attr_raw_npz_folder, "chunk_id-{}.npz".format(chunkid))
        ):
            logger.info(
                "Generation error: Not saving attr_raw data: {} not exists".format(
                    os.path.join(attr_raw_npz_folder, "chunk_id-{}.npz".format(chunkid))
                )
            )
            continue

        raw_attr_chunk = np.load(
            os.path.join(attr_raw_npz_folder, "chunk_id-{}.npz".format(chunkid))
        )["data_attribute"]

        if num_chunks > 1:
            for row in raw_attr_chunk:
                if (
                    row[col_idx_of_session_start] < row[col_idx_of_session_start + 1]
                    and row[col_idx_of_session_start + 2 * chunkid + 2]
                    < row[col_idx_of_session_start + 2 * chunkid + 3]
                ):
                    # this chunk
                    dict_chunkid_attr[chunkid].append(row)

                    # following chunks
                    n_flows_startFromThisEpoch += 1
                    row_following_chunk = list(copy.deepcopy(row))
                    row_following_chunk[col_idx_of_session_start] = 1.0
                    row_following_chunk[col_idx_of_session_start + 1] = 0.0

                    for i in range(chunkid + 1, num_chunks):
                        if (
                            row[col_idx_of_session_start + 2 * i + 2]
                            < row[col_idx_of_session_start + 2 * i + 3]
                        ):
                            dict_chunkid_attr[i].append(row_following_chunk)
        else:
            dict_chunkid_attr[chunkid] = raw_attr_chunk

    n_merged_attrs = 0
    for chunkid, attr_clean in dict_chunkid_attr.items():
        n_merged_attrs += len(attr_clean)
        np.savez(
            os.path.join(attr_clean_npz_folder, "chunk_id-{}.npz".format(chunkid)),
            data_attribute=np.asarray(attr_clean),
        )
