from .embedding_helper import build_annoy_dictionary_word2vec, get_original_obj
from netshare.utils import ContinuousField, DiscreteField, BitField
from netshare.utils import Normalization
from netshare.utils import Tee, Output
from scipy.spatial import distance
from scipy.stats import wasserstein_distance
from collections import Counter, OrderedDict
from gensim.models import Word2Vec
from tqdm import tqdm
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import sys
import configparser
import json
import random
import copy
import math
import os
import pickle
random.seed(42)


# avoid type3 fonts
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams.update({'font.size': 15})

# color-blindness friendly
CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']
# colors = {
#     'blue':   [55,  126, 184],  #377eb8
#     'orange': [255, 127, 0],    #ff7f00
#     'green':  [77,  175, 74],   #4daf4a
#     'pink':   [247, 129, 191],  #f781bf
#     'brown':  [166, 86,  40],   #a65628
#     'purple': [152, 78,  163],  #984ea3
#     'gray':   [153, 153, 153],  #999999
#     'red':    [228, 26,  28],   #e41a1c
#     'yellow': [222, 222, 0]     #dede00
# }

# https://www.iana.org/assignments/protocol-numbers/protocol-numbers.xhtml
dict_pr_str2int = {
    "ESP": 50,
    "GRE": 47,
    "ICMP": 1,
    "IPIP": 4,
    "IPv6": 41,
    "TCP": 6,
    "UDP": 17,
    "RSVP": 46,
    "Other": 255,
    "255": 255,  # TEMP
}


# jsd
def jsd(p, q, type):
    p = list(p)
    q = list(q)

    if type == "discrete":
        # append 0 to shorter arrays: only for IP
        pq_max_len = max(len(p), len(q))
        p += [0.0] * (pq_max_len - len(p))
        q += [0.0] * (pq_max_len - len(q))
        assert (len(p) == len(q))
        return distance.jensenshannon(p, q)**2

    elif type == "continuous":
        # min_ = min(min(p), min(q))
        # max_ = max(max(p), max(q))

        min_ = min(p)
        max_ = max(p)

        # assume p is raw data
        # compute n_bins by FD on raw data; use across baselines
        p_counts, p_bin_edges = np.histogram(
            p, range=(min_, max_), bins="auto")
        q_counts, q_bin_edges = np.histogram(
            q, range=(min_, max_), bins=len(p_counts))

        # out of range
        q_arr = np.array(q)
        q_arr_lt_realmin = q_arr[q_arr < min_]
        q_arr_gt_realmax = q_arr[q_arr > max_]

        if len(q_arr_lt_realmin) > 0:
            np.insert(q_counts, 0, len(q_arr_lt_realmin))
            np.insert(p_counts, 0, 0.0)
        if len(q_arr_gt_realmax) > 0:
            np.append(q_counts, len(q_arr_gt_realmax))
            np.append(p_counts, 0.0)

        return distance.jensenshannon(p_counts, q_counts)**2

    else:
        raise ValueError("Unknown JSD data type")


def compute_IP_rank_distance(real_list, syn_list, type="EMD"):
    real_HH_count = OrderedDict(Counter(real_list).most_common())
    syn_HH_count = OrderedDict(Counter(syn_list).most_common())

    real_rank_list = []
    idx = 1
    for k, v in real_HH_count.items():
        real_rank_list += [idx] * v
        idx += 1

    syn_rank_list = []
    idx = 1
    for k, v in syn_HH_count.items():
        syn_rank_list += [idx] * v
        idx += 1

    if type == "EMD":
        return wasserstein_distance(real_rank_list, syn_rank_list)
    elif type == "JSD":
        return jsd(real_HH_count.values(),
                   syn_HH_count.values(), type="discrete")
    else:
        raise ValueError("Unknown distance metric!")

# type == "freq": return the freq dict


def compute_port_proto_distance(
        real_list, syn_list, opt, prstr_raw=True, prstr_syn=True, type="TV"):
    real_list = list(real_list)
    syn_list = list(syn_list)

    # TCP: 6
    # UDP: 17
    # Other: 255, used for binning other protocols
    if opt == "proto":
        # convert to integer if protocol is string (e.g., "TCP"/"UDP")
        if isinstance(real_list[0], str):
            real_list_numeric = []
            for i in real_list:
                i = i.strip()
                real_list_numeric.append(dict_pr_str2int[i.upper()])
            real_list = real_list_numeric

        if isinstance(syn_list[0], str):
            syn_list_numeric = []
            for i in syn_list:
                i = i.strip()
                syn_list_numeric.append(dict_pr_str2int[i.upper()])
            syn_list = syn_list_numeric

    if opt == "srcport" or opt == "dstport":
        real_dict = {}
        syn_dict = {}
        for i in range(65536):
            real_dict[i] = 0
            syn_dict[i] = 0
        for i in real_list:
            real_dict[int(i)] += float(1 / len(real_list))
        for i in syn_list:
            if i < 0:
                i = 0
            elif i > 65535:
                i = 65535
            syn_dict[int(i)] += float(1 / len(syn_list))

        if type == "TV":
            tv_distance = 0
            for i in range(65536):
                tv_distance += 0.5 * abs(real_dict[i] - syn_dict[i])
            return tv_distance
        elif type == "JSD":
            return jsd(real_dict.values(), syn_dict.values(), type="discrete")
        elif type == "freq":
            return real_dict, syn_dict
        else:
            raise ValueError("Unknown distance metric!")

    elif opt == "proto":
        real_dict = {}
        syn_dict = {}
        for i in range(256):
            real_dict[i] = 0
            syn_dict[i] = 0
        for i in real_list:
            real_dict[int(i)] += float(1 / len(real_list))
        for i in syn_list:
            syn_dict[int(i)] += float(1 / len(syn_list))

        if type == "TV":
            tv_distance = 0
            for i in range(256):
                tv_distance += 0.5 * abs(real_dict[i] - syn_dict[i])
            return tv_distance
        elif type == "JSD":
            return jsd(real_dict.values(), syn_dict.values(), type="discrete")
        elif type == "freq":
            return real_dict, syn_dict
        else:
            raise ValueError("Unknown distance metric!")


def get_flowduration(df):
    df = df.sort_values("time")

    metadata = ["srcip", "dstip", "srcport", "dstport", "proto"]
    gk = df.groupby(by=metadata)

    flow_duration_list = []

    for name, group in gk:
        time_list = list(group["time"])
        flow_duration_list.append(time_list[-1] - time_list[0])

    return flow_duration_list


def compute_metrics_netflow_v3(raw_df, syn_df):
    '''JSD + EMD + ranking'''
    metrics_dict = {}

    # IP popularity rank
    for metric in ["srcip", "dstip"]:
        metrics_dict[metric] = compute_IP_rank_distance(
            raw_df[metric], syn_df[metric], type="JSD")

    # TV distance for port/protocol
    for metric in ["srcport", "dstport", "proto"]:
        metrics_dict[metric] = compute_port_proto_distance(
            raw_df[metric],
            syn_df[metric],
            metric, prstr_raw=True, prstr_syn=True, type="JSD")

    # ts, td, pkt, byt
    for metric in ["ts", "td", "pkt", "byt"]:
        if metric == "ts":
            raw_df = raw_df.sort_values("ts").reset_index()
            syn_df = syn_df.sort_values("ts").reset_index()
            raw_list = list(raw_df["ts"] - raw_df["ts"][0])
            syn_list = list(syn_df["ts"] - syn_df["ts"][0])
            metrics_dict[metric] = wasserstein_distance(raw_list, syn_list)
        else:
            metrics_dict[metric] = wasserstein_distance(
                list(raw_df[metric]), list(syn_df[metric]))

    return metrics_dict


def compute_metrics_zeeklog_v3(raw_df, syn_df):
    '''JSD + EMD + ranking'''
    metrics_dict = {}

    # IP popularity rank
    for metric in ["srcip", "dstip"]:
        metrics_dict[metric] = compute_IP_rank_distance(
            raw_df[metric], syn_df[metric], type="JSD")

    # TV distance for port/protocol
    for metric in ["srcport", "dstport", "proto"]:
        metrics_dict[metric] = compute_port_proto_distance(
            raw_df[metric],
            syn_df[metric],
            metric, prstr_raw=True, prstr_syn=True, type="JSD")

    # ts,duration,orig_bytes,resp_bytes,missed_bytes,orig_pkts,
    # orig_ip_bytes,resp_pkts,resp_ip_bytes
    for metric in ["ts", "duration", "orig_bytes", "resp_bytes", "missed_bytes",
                   "orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes"]:
        if metric == "ts":
            raw_df = raw_df.sort_values("ts").reset_index()
            syn_df = syn_df.sort_values("ts").reset_index()
            raw_list = list(raw_df["ts"] - raw_df["ts"][0])
            syn_list = list(syn_df["ts"] - syn_df["ts"][0])
            metrics_dict[metric] = wasserstein_distance(raw_list, syn_list)
        else:
            metrics_dict[metric] = wasserstein_distance(
                list(raw_df[metric]), list(syn_df[metric]))

    # TODO: Important!! How to define the JSD of service and conn_state?

    return metrics_dict


def compute_metrics_pcap_v3(raw_df, syn_df):
    '''JSD + EMD + ranking'''
    metrics_dict = {}

    # IP popularity rank
    for metric in ["srcip", "dstip"]:
        metrics_dict[metric] = compute_IP_rank_distance(
            raw_df[metric], syn_df[metric], type="JSD")

    # TV distance for port/protocol
    for metric in ["srcport", "dstport", "proto"]:
        metrics_dict[metric] = compute_port_proto_distance(
            raw_df[metric],
            syn_df[metric],
            metric, prstr_raw=True, prstr_syn=True, type="JSD")

    # pkt_len
    for metric in ["pkt_len", "time"]:
        # if metric == "time":
        #     label = "pkt_arrivalTime"
        # else:
        #     label = metric

        if metric == "time":
            raw_df = raw_df.sort_values("time").reset_index()
            syn_df = syn_df.sort_values("time").reset_index()
            raw_list = list(raw_df["time"] - raw_df["time"][0])
            syn_list = list(syn_df["time"] - syn_df["time"][0])
            metrics_dict[metric] = wasserstein_distance(raw_list, syn_list)
        else:
            metrics_dict[metric] = wasserstein_distance(
                list(raw_df[metric]), list(syn_df[metric]))

    # interarrival time
    # raw_df = raw_df.sort_values("time")
    # syn_df = syn_df.sort_values("time")
    # metrics_dict["PIAT"] = wasserstein_distance(list(np.diff(raw_df["time"])), list(np.diff(syn_df["time"])))

    # flow size distribution
    metadata = ["srcip", "dstip", "srcport", "dstport", "proto"]
    raw_gk = raw_df.groupby(by=metadata)
    syn_gk = syn_df.groupby(by=metadata)

    raw_flowsize_list = list(raw_gk.size().values)
    syn_flowsize_list = list(syn_gk.size().values)
    metrics_dict["flow_size"] = wasserstein_distance(
        raw_flowsize_list, syn_flowsize_list)

    return metrics_dict
