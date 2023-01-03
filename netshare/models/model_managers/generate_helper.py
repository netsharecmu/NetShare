import copy
import os
from pathlib import Path
from typing import Dict, List, Type

import numpy as np
import pandas as pd
from tqdm import tqdm

import netshare.utils.ray as ray
from netshare.models import Model
from netshare.utils.logger import logger
from netshare.utils.paths import get_generated_data_log_folder


@ray.remote(scheduling_strategy="SPREAD", max_calls=1)
def generate_data(
    create_new_model: Type[Model],
    config: dict,
    given_data_attribute_flag: bool,
) -> None:
    config["given_data_attribute_flag"] = given_data_attribute_flag
    model = create_new_model(config)
    model.generate(
        input_train_data_folder=config["dataset"],
        input_model_folder=config["result_folder"],
        output_syn_data_folder=config["eval_root_folder"],
        log_folder=get_generated_data_log_folder(),
    )


@ray.remote(scheduling_strategy="SPREAD", max_calls=1)
def merge_attr(config_group: dict, configs: List[dict]) -> None:
    """
    TODO: Can someone help me to doc this function?
    """
    chunk0_idx = config_group["config_ids"][0]
    eval_root_folder = configs[chunk0_idx]["eval_root_folder"]
    attr_raw_npz_folder = os.path.join(eval_root_folder, "attr_raw")
    word2vec_size = configs[chunk0_idx].get("word2vec_vecSize", 0)
    pcap_interarrival = configs[chunk0_idx].get("timestamp") == "interarrival"
    num_chunks = len(config_group["config_ids"])

    if not pcap_interarrival:
        bit_idx_flagstart = 128 + word2vec_size * 3
    else:
        bit_idx_flagstart = 128 + word2vec_size * 3 + 1

    attr_clean_npz_folder = os.path.join(
        str(Path(attr_raw_npz_folder).parents[0]), "attr_clean"
    )
    os.makedirs(attr_clean_npz_folder, exist_ok=True)

    dict_chunkid_attr: Dict[int, List[List[float]]] = {}
    for chunkid in range(num_chunks):
        dict_chunkid_attr[chunkid] = []

    for chunkid in tqdm(range(num_chunks)):
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

        if num_chunks > 1 and word2vec_size:
            for row in raw_attr_chunk:
                # if row[bit_idx_flagstart] < row[bit_idx_flagstart+1]:
                if (
                    row[bit_idx_flagstart] < row[bit_idx_flagstart + 1]
                    and row[bit_idx_flagstart + 2 * chunkid + 2]
                    < row[bit_idx_flagstart + 2 * chunkid + 3]
                ):
                    # this chunk
                    row_this_chunk = list(copy.deepcopy(row)[:bit_idx_flagstart])
                    row_this_chunk += [0.0, 1.0]
                    row_this_chunk += [1.0, 0.0] * (chunkid + 1)
                    for i in range(chunkid + 1, num_chunks):
                        if (
                            row[bit_idx_flagstart + 2 * i + 2]
                            < row[bit_idx_flagstart + 2 * i + 3]
                        ):
                            row_this_chunk += [0.0, 1.0]
                        else:
                            row_this_chunk += [1.0, 0.0]
                    # dict_chunkid_attr[chunkid].append(row_this_chunk)
                    dict_chunkid_attr[chunkid].append(row)

                    # following chunks
                    # row_following_chunk = list(copy.deepcopy(row)[:bit_idx_flagstart])
                    # row_following_chunk += [1.0, 0.0]*(1+NUM_CHUNKS)
                    n_flows_startFromThisEpoch += 1
                    row_following_chunk = list(copy.deepcopy(row))
                    row_following_chunk[bit_idx_flagstart] = 1.0
                    row_following_chunk[bit_idx_flagstart + 1] = 0.0

                    for i in range(chunkid + 1, num_chunks):
                        if (
                            row[bit_idx_flagstart + 2 * i + 2]
                            < row[bit_idx_flagstart + 2 * i + 3]
                        ):
                            dict_chunkid_attr[i].append(row_following_chunk)
                            # dict_chunkid_attr[i].append(row)
        else:
            dict_chunkid_attr[chunkid] = raw_attr_chunk

        logger.debug(
            "n_flows_startFromThisEpoch / total flows: {}/{}".format(
                n_flows_startFromThisEpoch, raw_attr_chunk.shape[0]
            )
        )

    logger.debug("Saving merged attributes")
    n_merged_attrs = 0
    for chunkid, attr_clean in dict_chunkid_attr.items():
        n_merged_attrs += len(attr_clean)
        np.savez(
            os.path.join(attr_clean_npz_folder, "chunk_id-{}.npz".format(chunkid)),
            data_attribute=np.asarray(attr_clean),
        )


# ===================== TODO: move merge_syn_df to postprocess ================


def get_per_chunk_df(chunk_folder: str) -> pd.DataFrame:
    '''chunk_folder: "chunk_id-0"'''
    df_names = [file for file in os.listdir(chunk_folder) if file.endswith(".csv")]
    assert len(df_names) > 0
    df = pd.read_csv(os.path.join(chunk_folder, df_names[0]))

    return df
