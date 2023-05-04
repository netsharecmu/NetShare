import copy
import subprocess
import sys
import time
import os
import json
import importlib
import random
import pickle
import pandas as pd
import socket
import struct
import ipaddress
import argparse

import numpy as np
import pandas as pd

import netshare.ray as ray
from pathlib import Path
from tqdm import tqdm
from scapy.all import IP, ICMP, TCP, UDP
from scapy.all import wrpcap
from scipy.stats import rankdata
from pathlib import Path


@ray.remote(scheduling_strategy="SPREAD", max_calls=1)
def _generate_session(
        create_new_model,
        configs,
        config_idx,
        log_folder):
    config = configs[config_idx]
    config["given_data_attribute_flag"] = False
    model = create_new_model(config)
    model.generate(
        input_train_data_folder=config["dataset"],
        input_model_folder=config["result_folder"],
        output_syn_data_folder=config["eval_root_folder"],
        log_folder=log_folder)


@ray.remote(scheduling_strategy="SPREAD", max_calls=1)
def _generate_attr(
        create_new_model,
        configs,
        config_idx,
        log_folder):
    config = configs[config_idx]
    config["given_data_attribute_flag"] = False
    model = create_new_model(config)
    model.generate(
        input_train_data_folder=config["dataset"],
        input_model_folder=config["result_folder"],
        output_syn_data_folder=config["eval_root_folder"],
        log_folder=log_folder)


@ray.remote(scheduling_strategy="SPREAD", max_calls=1)
def _merge_attr(
    attr_raw_npz_folder,
    config_group,
    configs
):
    num_chunks = len(config_group["config_ids"])
    chunk0_idx = config_group["config_ids"][0]
    chunk0_config = configs[chunk0_idx]
    print("chunk0 config:", configs[chunk0_idx])

    # Find flow tag starting point
    with open(os.path.join(chunk0_config["dataset"], "data_attribute_fields.pkl"), 'rb') as f:
        data_attribute_fields = pickle.load(f)
    bit_idx_flagstart = 0
    for field_idx, field in enumerate(data_attribute_fields):
        if field.name != "startFromThisChunk":
            bit_idx_flagstart += field.dim_x
        else:
            break
    print("bit_idx_flagstart:", bit_idx_flagstart)

    attr_clean_npz_folder = os.path.join(
        str(Path(attr_raw_npz_folder).parents[0]), "attr_clean"
    )
    os.makedirs(attr_clean_npz_folder, exist_ok=True)

    dict_chunkid_attr = {}
    dict_chunkid_attr_discrete = {}
    for chunkid in tqdm(range(num_chunks)):
        dict_chunkid_attr[chunkid] = []
        dict_chunkid_attr_discrete[chunkid] = []

    for chunkid in tqdm(range(num_chunks)):
        n_flows_startFromThisEpoch = 0

        if not os.path.exists(
            os.path.join(
                attr_raw_npz_folder,
                "chunk_id-{}.npz".format(chunkid))
        ):
            print(
                "{} not exists...".format(
                    os.path.join(
                        attr_raw_npz_folder,
                        "chunk_id-{}.npz".format(chunkid))
                )
            )
            continue

        raw_attr_chunk = np.load(
            os.path.join(
                attr_raw_npz_folder,
                "chunk_id-{}.npz".format(chunkid))
        )["data_attribute"]
        raw_attr_discrete_chunk = np.load(
            os.path.join(
                attr_raw_npz_folder,
                "chunk_id-{}.npz".format(chunkid))
        )["data_attribute_discrete"]

        if num_chunks > 1:
            for row_idx, row in enumerate(raw_attr_chunk):
                # if row[bit_idx_flagstart] < row[bit_idx_flagstart+1]:
                if (
                    row[bit_idx_flagstart] < row[bit_idx_flagstart + 1]
                    and row[bit_idx_flagstart + 2 * chunkid + 2]
                    < row[bit_idx_flagstart + 2 * chunkid + 3]
                ):
                    # this chunk
                    row_this_chunk = list(
                        copy.deepcopy(row)[
                            :bit_idx_flagstart])
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
                    dict_chunkid_attr_discrete[chunkid].append(
                        raw_attr_discrete_chunk[row_idx])

                    # following chunks
                    # row_following_chunk = list(copy.deepcopy(row)[:bit_idx_flagstart])
                    # row_following_chunk += [1.0, 0.0]*(1+NUM_CHUNKS)
                    n_flows_startFromThisEpoch += 1
                    row_following_chunk = list(copy.deepcopy(row))
                    row_following_chunk[bit_idx_flagstart] = 1.0
                    row_following_chunk[bit_idx_flagstart + 1] = 0.0

                    row_discrete_following_chunk = list(
                        copy.deepcopy(raw_attr_discrete_chunk[row_idx]))
                    row_discrete_following_chunk[bit_idx_flagstart] = 1.0
                    row_discrete_following_chunk[bit_idx_flagstart + 1] = 0.0

                    for i in range(chunkid + 1, num_chunks):
                        if (
                            row[bit_idx_flagstart + 2 * i + 2]
                            < row[bit_idx_flagstart + 2 * i + 3]
                        ):
                            dict_chunkid_attr[i].append(row_following_chunk)
                            dict_chunkid_attr_discrete[i].append(
                                row_discrete_following_chunk)
                            # dict_chunkid_attr[i].append(row)
        else:
            dict_chunkid_attr[chunkid] = raw_attr_chunk
            dict_chunkid_attr_discrete[chunkid] = raw_attr_discrete_chunk

        print(
            "n_flows_startFromThisEpoch / total flows: {}/{}".format(
                n_flows_startFromThisEpoch, raw_attr_chunk.shape[0]
            )
        )

    print("Saving merged attrs...")
    n_merged_attrs = 0
    for chunkid, attr_clean in dict_chunkid_attr.items():
        print("chunk {}: {} flows".format(chunkid, len(attr_clean)))
        n_merged_attrs += len(attr_clean)
        np.savez(
            os.path.join(
                attr_clean_npz_folder, "chunk_id-{}.npz".format(chunkid)),
            data_attribute=np.asarray(attr_clean),
            data_attribute_discrete=np.asarray(
                dict_chunkid_attr_discrete[chunkid]))

    print("n_merged_attrs:", n_merged_attrs)


@ray.remote(scheduling_strategy="SPREAD", max_calls=1)
# @ray.remote(scheduling_strategy="DEFAULT", max_calls=1)
def _generate_given_attr(create_new_model, configs, config_idx,
                         log_folder):

    config = configs[config_idx]
    config["given_data_attribute_flag"] = True
    model = create_new_model(config)
    model.generate(
        input_train_data_folder=config["dataset"],
        input_model_folder=config["result_folder"],
        output_syn_data_folder=config["eval_root_folder"],
        log_folder=log_folder)
