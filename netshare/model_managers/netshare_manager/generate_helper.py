import copy
import subprocess
import sys
import time
import os
import json
import importlib
import random
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

from ...pre_post_processors.netshare.dist_metrics import compute_metrics_netflow_v3, compute_metrics_pcap_v3


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
def _merge_attr(attr_raw_npz_folder, word2vec_size,
                pcap_interarrival, num_chunks):
    if not pcap_interarrival:
        bit_idx_flagstart = 128 + word2vec_size * 3
    else:
        bit_idx_flagstart = 128 + word2vec_size * 3 + 1

    print("PCAP_INTERARRIVAL:", pcap_interarrival)
    print("bit_idx_flagstart:", bit_idx_flagstart)

    attr_clean_npz_folder = os.path.join(
        str(Path(attr_raw_npz_folder).parents[0]), "attr_clean"
    )
    os.makedirs(attr_clean_npz_folder, exist_ok=True)

    dict_chunkid_attr = {}
    for chunkid in tqdm(range(num_chunks)):
        dict_chunkid_attr[chunkid] = []

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
                attr_clean_npz_folder,
                "chunk_id-{}.npz".format(chunkid)),
            data_attribute=np.asarray(attr_clean),
        )

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

# ===================== TODO: move merge_syn_df to postprocess ================


def IP_int2str(IP_int):
    return str(ipaddress.ip_address(IP_int))


def IP_str2int(IP_str):
    return int(ipaddress.ip_address(IP_str))


def IPs_int2str(IPs_int):
    return [IP_int2str(i) for i in IPs_int]


def IPs_str2int(IPs_str):
    return [IP_str2int(i) for i in IPs_str]


pr_dict = {
    "ESP": 50,
    "GRE": 47,
    "ICMP": 1,
    "IPIP": 4,
    "IPv6": 41,
    "TCP": 6,
    "UDP": 17,
    "Other": 255
}


def prs_str2int(prs):
    prs_int = []
    for p in prs:
        prs_int.append(pr_dict[p])
    return prs_int


pr_int2str_dict = {
    1: "ICMP",
    4: "IPIP",
    6: "TCP",
    17: "UDP",
    41: "IPv6",
    47: "GRE",
    50: "ESP",
    255: "Other"
}


def prs_int2str(prs_int):
    prs_str = []
    for p in prs_int:
        prs_str.append(pr_int2str_dict[p])
    return prs_str


def csv2pcap_single(input, output):
    # df = pd.read_csv(input).sort_values(["time"])
    df = input.sort_values(["time"])

    packets = []

    for i, row in tqdm(df.iterrows(), total=df.shape[0]):
        time = float(row["time"] / 10**6)
        if isinstance(row["srcip"], str):
            srcip = IP_str2int(row["srcip"])
            dstip = IP_str2int(row["dstip"])
            src = socket.inet_ntoa(struct.pack('!L', srcip))
            dst = socket.inet_ntoa(struct.pack('!L', dstip))
        else:
            src = socket.inet_ntoa(struct.pack('!L', row["srcip"]))
            dst = socket.inet_ntoa(struct.pack('!L', row["dstip"]))

        srcport = row["srcport"]
        dstport = row["dstport"]
        proto = row["proto"]
        pkt_len = int(row["pkt_len"])

        try:
            proto = int(proto)
        except:
            if proto == "TCP":
                proto = 6
            elif proto == "UDP":
                proto = 17
            elif proto == "ICMP":
                proto = 1
            else:
                proto = 0

        ip = IP(src=src, dst=dst, len=pkt_len, proto=proto)
        if proto == 1:
            p = ip / ICMP()
        elif proto == 6:
            tcp = TCP(sport=srcport, dport=dstport)
            p = ip / tcp
        elif proto == 17:
            udp = UDP(sport=srcport, dport=dstport)
            p = ip / udp
        else:
            p = ip

        p.time = time
        p.len = pkt_len
        p.wirelen = pkt_len + 4

        packets.append(p)

    wrpcap(output, packets)


def get_per_chunk_df(chunk_folder):
    '''chunk_folder: "chunk_id-0"'''
    df_names = [file for file in os.listdir(
        chunk_folder) if file.endswith(".csv")]
    assert (len(df_names) > 0)
    df = pd.read_csv(os.path.join(chunk_folder, df_names[0]))

    return df


def compare_rawdf_syndfs(raw_df, syn_dfs, data_type):
    metrics_dict_list = []
    for syn_df in syn_dfs:
        if data_type == "pcap":
            metrics_dict = compute_metrics_pcap_v3(raw_df, syn_df)
        elif data_type == "netflow":
            syn_df["pkt"] = np.round(syn_df["pkt"])
            syn_df["byt"] = np.round(syn_df["byt"])
            metrics_dict = compute_metrics_netflow_v3(raw_df, syn_df)
        else:
            raise ValueError("Unknown data type! Must be netflow or pcap...")
        metrics_dict_list.append(metrics_dict)

#     print(metrics_dict_list)
    metrics = list(metrics_dict_list[0].keys())
    # print(metrics)

    metric_vals_dict = {}
    for metrics_dict in metrics_dict_list:
        for metric in metrics:
            if metric not in metric_vals_dict:
                metric_vals_dict[metric] = []
            metric_vals_dict[metric].append(metrics_dict[metric])

    metric_vals_2d = []
    for metric, vals in metric_vals_dict.items():
        metric_vals_2d.append(vals)

#     print(metric_vals_2d)
    rankings_sum = np.sum(rankdata(metric_vals_2d, axis=1), axis=0)
    # print(rankings_sum)
#     print(rankdata(rankings_sum))
#     print(np.argmin(rankdata(rankings_sum)))
    best_syndf_idx = np.argmin(rankdata(rankings_sum))

    return best_syndf_idx, syn_dfs[best_syndf_idx]


def last_lvl_folder(folder):
    return str(Path(folder).parents[0])


def _merge_syn_df(
        configs,
        config_group_list,
        big_raw_df,
        output_syn_data_folder):
    '''TODO: CHANGE TO DISTRIBUTE'''
    # (dataset, dp_noise_multiplier) : best_syndf_chunki
    dict_dataset_syndfs = {}
    for config_group_idx, config_group in enumerate(config_group_list):
        print("Config group #{}: {}".format(config_group_idx, config_group))
        config_ids = config_group["config_ids"]
        chunk0_idx = config_ids[0]
        syndf_root_folder = os.path.join(
            configs[chunk0_idx]["eval_root_folder"],
            "syn_dfs"
        )
        assert len([file for file in os.listdir(syndf_root_folder)
                   if file.startswith("chunk_id")]) == len(config_ids)

        best_syndfs = []
        for chunk_id, config_idx in enumerate(config_ids):
            config = configs[config_idx]
            raw_df = pd.read_csv(os.path.join(
                config["dataset"],
                "raw.csv"
            ))

            if config["dataset_type"] == "pcap":
                data_type = "pcap"
                time_col_name = "time"
            elif config["dataset_type"] == "netflow":
                data_type = "netflow"
                time_col_name = "ts"
            else:
                raise ValueError(
                    "Unknown data type! Must be netflow or pcap...")

            syn_dfs = []
            syn_dfs_names = []
            syn_df_folder = os.path.join(
                syndf_root_folder,
                "chunk_id-{}".format(chunk_id)
            )
            for file in os.listdir(syn_df_folder):
                if file.endswith(".csv"):
                    syn_dfs_names.append(file)
                    syn_df = pd.read_csv(os.path.join(syn_df_folder, file))

                    # truncate to raw data time range
                    if config["truncate"] == "per_chunk":
                        syn_df = syn_df[(syn_df[time_col_name] >= raw_df[time_col_name].min()) & (
                            syn_df[time_col_name] <= raw_df[time_col_name].max())]
                    elif config["truncate"] == "none":
                        pass
                    # TODO: support more truncation methods if necessary
                    else:
                        raise ValueError("Unknown truncation methods...")

                    syn_dfs.append(syn_df)

            best_syndf_idx, best_syndf = compare_rawdf_syndfs(
                raw_df, syn_dfs, data_type)
            best_syndfs.append(best_syndf)

            print("Chunk_id: {}, # of syn dfs: {}, best_syndf: {}".format(
                chunk_id, len(syn_dfs), syn_dfs_names[best_syndf_idx]))

        big_best_syndf = pd.concat(best_syndfs)
        print("Big syndf shape:", big_best_syndf.shape)
        print()

        if config_group["dp_noise_multiplier"] not in dict_dataset_syndfs:
            dict_dataset_syndfs[config_group["dp_noise_multiplier"]] = []
        dict_dataset_syndfs[config_group["dp_noise_multiplier"]].append(
            big_best_syndf)

    dict_dataset_bestsyndf = {}
    for dpnoisemultiplier, syn_dfs in dict_dataset_syndfs.items():
        assert len(syn_dfs) >= 1
        if len(syn_dfs) > 1:
            best_syndf_idx, best_syn_df = compare_rawdf_syndfs(
                big_raw_df, syn_dfs, data_type)
            dict_dataset_bestsyndf[dpnoisemultiplier] = best_syn_df
        else:
            dict_dataset_bestsyndf[dpnoisemultiplier] = syn_dfs[0]

    print("Aggregated final dataset syndf")
    for dp_noise_multiplier, best_syndf in dict_dataset_bestsyndf.items():
        print(dp_noise_multiplier, best_syndf.shape)
        best_syndf_folder = os.path.join(
            output_syn_data_folder,
            "best_syn_dfs"
        )
        os.makedirs(best_syndf_folder, exist_ok=True)

        # find best syndf index i.e., for evaluation fairness
        cur_max_idx = None
        for file in os.listdir(best_syndf_folder):
            if file.startswith("syn_df,dp_noise_multiplier-{},truncate-{},id-".format(dp_noise_multiplier, config["truncate"])):
                this_id = int(os.path.splitext(file)[
                              0].split(",")[-1].split("-")[1])
                if cur_max_idx == None or this_id > cur_max_idx:
                    cur_max_idx = this_id
        if cur_max_idx == None:
            cur_max_idx = 1
        else:
            cur_max_idx += 1

        # best_syndf_filename = os.path.join(
        #     best_syndf_folder,
        #     "syn_df,dp_noise_multiplier-{},truncate-{},id-{}.csv".format(
        #         dp_noise_multiplier, config["truncate"], cur_max_idx)
        # )
        best_syndf_filename = os.path.join(
            best_syndf_folder,
            "syn.csv"
        )

        print("best_syn_df filename:", best_syndf_filename)

        # sort by timestamp
        if configs[0]["dataset_type"] == "pcap":
            time_col_name = "time"
        elif configs[0]["dataset_type"] == "netflow":
            time_col_name = "ts"
        else:
            raise ValueError("Unknown data type! Must be netflow or pcap...")

        best_syndf = best_syndf.sort_values(time_col_name)
        best_syndf.to_csv(best_syndf_filename, index=False)

        if configs[0]["dataset_type"] == "pcap":
            best_synpcap_filename = os.path.join(
                best_syndf_folder,
                "syn.pcap"
            )
            csv2pcap_single(best_syndf, best_synpcap_filename)
