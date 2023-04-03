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


data_baseDir = "../data/1M"
result_baseDir = "../results/results_sigcomm2022_slides"

N_TOPK_SERVICE_PORTS = 5


def vals2cdf(vals):
    dist_dict = dict(Counter(vals))
    dist_dict = {k: v for k, v in sorted(
        dist_dict.items(), key=lambda x: x[0])}
    x = dist_dict.keys()

    pdf = np.asarray(list(dist_dict.values()), dtype=float) / \
        float(sum(dist_dict.values()))
    cdf = np.cumsum(pdf)

    return x, cdf


# syn_df_dict: {name: dict}
def plot_cdf(raw_df, syn_df_dict, xlabel, ylabel, plot_loc, metric,
             x_logscale=False, y_logscale=False):
    plt.clf()

    if metric == "flow_size":
        x, cdf = vals2cdf(raw_df.groupby(
            ["srcip", "dstip", "srcport", "dstport", "proto"]).size().values)
    else:
        x, cdf = vals2cdf(raw_df[metric])

    plt.plot(x, cdf, label="Real", color=CB_color_cycle[0], linewidth=5)
    idx = 1
    for method, syn_df in syn_df_dict.items():
        if method == "CTGAN-B":
            label_method = "CTGAN"
        else:
            label_method = method

        if metric == "pkt" or metric == "byt":
            syn_df[metric] = np.round(syn_df[metric])

        if metric == "flow_size":
            x, cdf = vals2cdf(syn_df.groupby(
                ["srcip", "dstip", "srcport", "dstport", "proto"]).size().values)
        elif metric == "td":
            x, cdf = vals2cdf(np.round(syn_df[metric]))
        else:
            x, cdf = vals2cdf(syn_df[metric])

        if method == "NetShare":
            plt.plot(x, cdf, label=label_method,
                     color=CB_color_cycle[4], linewidth=3)

        else:
            plt.plot(x, cdf, label=label_method,
                     color=CB_color_cycle[idx], linewidth=1.5)

        idx += 1

    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.legend(fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    if x_logscale:
        plt.xscale('log')
    if y_logscale:
        plt.yscale('log')

    plt.savefig(plot_loc, bbox_inches="tight", dpi=300)


def get_HH_unordered(df, col_key):
    HH_count = OrderedDict(Counter(df[col_key]).most_common())
    HH_density = OrderedDict()
    for k, v in HH_count.items():
        HH_density[k] = v / float(len(df[col_key]))
    return [i+1 for i in range(len(HH_density))], list(HH_density.values())


def plot_HH(real_df, syn_df_dict, xlabel, ylabel, plot_loc, metric,
            x_logscale=False, y_logscale=False):
    plt.clf()

    x, y = get_HH_unordered(real_df, metric)
    plt.plot(x, y, label="Real", color=CB_color_cycle[0], linewidth=3)

    idx = 1
    for method, syn_df in syn_df_dict.items():
        x, y = get_HH_unordered(syn_df, metric)
        plt.plot(x, y, label=method, color=CB_color_cycle[idx], linewidth=1)
        idx += 1

    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.legend(fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    if x_logscale:
        plt.xscale('log')
    if y_logscale:
        plt.yscale('log')

    plt.savefig(plot_loc, bbox_inches="tight", dpi=300)

# real_dict: {port_number: count} port_number \in [0, ... 65535]
# RETURN: topN_service_ports


def get_common_service_port(real_dict, topN=10):
    service_ports = [i for i in range(1024)]
    service_ports_count = {k: real_dict[k] for k in service_ports}
    service_ports_count = {k: v for k, v in sorted(
        service_ports_count.items(), key=lambda x: x[1], reverse=True)}

    topN_service_ports = list(service_ports_count.keys())[:topN]
    # topN_service_ports = sorted(topN_service_ports)

    return topN_service_ports


def plot_bar(real_df, syn_df_dict, xlabel, ylabel, plot_loc, metric,
             x_logscale=False, y_logscale=False, data_type="netflow"):
    plt.clf()

    if metric == "srcport" or metric == "dstport":
        fig, axs = plt.subplots(len(syn_df_dict), 1, figsize=(8, 6))
        # fig, axs = plt.subplots(2, 2, figsize=(12, 7))
        bar_width = 0.4
    elif metric == "proto":
        fig, axs = plt.subplots(len(syn_df_dict), 1, figsize=(8, 6))
        bar_width = 0.2

    subplot_idx = 0
    for method, syn_df in syn_df_dict.items():
        if data_type == "netflow":
            real_dict, syn_dict = compute_port_proto_distance(
                real_df[metric],
                syn_df[metric],
                opt=metric, prstr_raw=True, prstr_syn=False, type="freq")
        elif data_type == "pcap":
            real_dict, syn_dict = compute_port_proto_distance(
                real_df[metric],
                syn_df[metric],
                opt=metric, prstr_raw=True, prstr_syn=False, type="freq")
        else:
            raise ValueError(
                "Non-valid data type! Must be either netflow or pcap")

        if metric == "srcport" or metric == "dstport":
            x = get_common_service_port(real_dict, topN=N_TOPK_SERVICE_PORTS)
            x_plot = [str(i) for i in x]
        # TCP, UDP, Others
        elif metric == "proto":
            x = [6, 17]
            x_plot = ["TCP", "UDP", "Others"]

        y_real = [real_dict[i] for i in x]
        y_syn = [syn_dict[i] for i in x]

        # other protocol
        if metric == "proto":
            y_real.append(1.0-y_real[0]-y_real[1])
            y_syn.append(1.0-y_syn[0]-y_syn[1])

        if len(syn_df_dict) == 1:
            ax = axs
        else:
            ax = axs[subplot_idx]

        ax.bar(x_plot, y_real, alpha=0.5, label="Real",
               color=CB_color_cycle[0], width=bar_width)
        ax.bar(x_plot, y_syn, alpha=0.5, label=method,
               color=CB_color_cycle[1], width=bar_width)
        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=18)

        subplot_idx += 1

        # ax.set_xlabel(xlabel, fontsize=14)
        # ax.set_ylabel(ylabel, fontsize=14)
        ax.legend(fontsize=20, loc="upper right")
        if x_logscale:
            ax.set_xscale('log')
        if y_logscale:
            ax.set_yscale('log')

    fig.text(0.5, 0.0, xlabel, ha="center", va="center", fontsize=20)
    fig.text(0.0, 0.5, ylabel, ha="center",
             va="center", fontsize=20, rotation=90)

    plt.savefig(plot_loc, bbox_inches="tight", dpi=300)


def plot_bar_port(
        real_df, syn_df_dict, xlabel, ylabel, plot_loc, metric,
        x_logscale=False, y_logscale=False, data_type="netflow"):
    plt.clf()

    if metric == "srcport" or metric == "dstport":
        # fig, axs = plt.subplots(len(syn_df_dict), 1, figsize=(8, 10))
        fig, axs = plt.subplots(2, 2, figsize=(12, 7))
        bar_width = 0.4
    elif metric == "proto":
        fig, axs = plt.subplots(len(syn_df_dict), 1, figsize=(6, 5))
        bar_width = 0.2

    subplot_idx = 0
    for method, syn_df in syn_df_dict.items():
        if data_type == "netflow":
            real_dict, syn_dict = compute_port_proto_distance(
                real_df[metric],
                syn_df[metric],
                opt=metric, prstr_raw=True, prstr_syn=False, type="freq")
        elif data_type == "pcap":
            real_dict, syn_dict = compute_port_proto_distance(
                real_df[metric],
                syn_df[metric],
                opt=metric, prstr_raw=True, prstr_syn=False, type="freq")
        else:
            raise ValueError(
                "Non-valid data type! Must be either netflow or pcap")

        if metric == "srcport" or metric == "dstport":
            x = get_common_service_port(real_dict, topN=N_TOPK_SERVICE_PORTS)
            x_plot = [str(i) for i in x]
        # TCP, UDP, Others
        elif metric == "proto":
            x = [6, 17]
            x_plot = ["TCP", "UDP", "Others"]

        y_real = [real_dict[i] for i in x]
        y_syn = [syn_dict[i] for i in x]

        # other protocol
        if metric == "proto":
            y_real.append(1.0-y_real[0]-y_real[1])
            y_syn.append(1.0-y_syn[0]-y_syn[1])

        if len(syn_df_dict) == 1:
            ax = axs
        else:
            if subplot_idx == 0:
                ax = axs[0][0]
            if subplot_idx == 1:
                ax = axs[0][1]
            if subplot_idx == 2:
                ax = axs[1][0]
            if subplot_idx == 3:
                ax = axs[1][1]

        ax.bar(x_plot, y_real, alpha=0.5, label="Real",
               color=CB_color_cycle[0], width=bar_width)

        if method == "CTGAN-B":
            method_label = "CTGAN"
        else:
            method_label = method

        ax.bar(x_plot, y_syn, alpha=0.5, label=method_label,
               color=CB_color_cycle[1], width=bar_width)
        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=18)

        subplot_idx += 1

        # ax.set_xlabel(xlabel, fontsize=14)
        # ax.set_ylabel(ylabel, fontsize=14)
        ax.legend(fontsize=22, loc="upper right")
        if x_logscale:
            ax.set_xscale('log')
        if y_logscale:
            ax.set_yscale('log')

    fig.text(0.5, 0.03, xlabel, ha="center", va="center", fontsize=24)
    fig.text(0.05, 0.5, ylabel, ha="center",
             va="center", fontsize=24, rotation=90)

    plt.savefig(plot_loc, bbox_inches="tight", dpi=300)


# jsd
def jsd(p, q, type):
    p = list(p)
    q = list(q)

    if type == "discrete":
        # append 0 to shorter arrays: only for IP
        pq_max_len = max(len(p), len(q))
        p += [0.0]*(pq_max_len - len(p))
        q += [0.0]*(pq_max_len - len(q))
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
        real_rank_list += [idx]*v
        idx += 1

    syn_rank_list = []
    idx = 1
    for k, v in syn_HH_count.items():
        syn_rank_list += [idx]*v
        idx += 1

    if type == "EMD":
        return wasserstein_distance(real_rank_list, syn_rank_list)
    elif type == "JSD":
        return jsd(
            real_HH_count.values(),
            syn_HH_count.values(),
            type="discrete")
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
                real_list_numeric.append(dict_pr_str2int[i])
            real_list = real_list_numeric

        if isinstance(syn_list[0], str):
            syn_list_numeric = []
            for i in syn_list:
                i = i.strip()
                syn_list_numeric.append(dict_pr_str2int[i])
            syn_list = syn_list_numeric

    if opt == "srcport" or opt == "dstport":
        real_dict = {}
        syn_dict = {}
        for i in range(65536):
            real_dict[i] = 0
            syn_dict[i] = 0
        for i in real_list:
            real_dict[int(i)] += float(1/len(real_list))
        for i in syn_list:
            if i < 0:
                i = 0
            elif i > 65535:
                i = 65535
            syn_dict[int(i)] += float(1/len(syn_list))

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
            real_dict[int(i)] += float(1/len(real_list))
        for i in syn_list:
            syn_dict[int(i)] += float(1/len(syn_list))

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


def compute_metrics_netflow(raw_df, syn_df):
    metrics_dict = {}

    # IP popularity rank
    for metric in ["srcip", "dstip"]:
        metrics_dict[metric] = compute_IP_rank_distance(
            raw_df[metric], syn_df[metric])

    # TV distance for port/protocol
    for metric in ["srcport", "dstport", "proto"]:
        metrics_dict[metric] = compute_port_proto_distance(
            raw_df[metric], syn_df[metric], metric)

    # ts, td, pkt, byt
    for metric in ["td", "pkt", "byt"]:
        metrics_dict[metric] = wasserstein_distance(
            list(raw_df[metric]), list(syn_df[metric]))

    return metrics_dict


def compute_metrics_pcap(raw_df, syn_df):
    metrics_dict = {}

    # IP popularity rank
    for metric in ["srcip", "dstip"]:
        metrics_dict[metric] = compute_IP_rank_distance(
            raw_df[metric], syn_df[metric])

    # TV distance for port/protocol
    for metric in ["srcport", "dstport", "proto"]:
        metrics_dict[metric] = compute_port_proto_distance(
            raw_df[metric], syn_df[metric], metric, prstr_raw=True, prstr_syn=False)

    # pkt_len
    for metric in ["pkt_len"]:
        metrics_dict[metric] = wasserstein_distance(
            list(raw_df[metric]), list(syn_df[metric]))

    # flow size distribution
    metadata = ["srcip", "dstip", "srcport", "dstport", "proto"]
    raw_gk = raw_df.groupby(by=metadata)
    syn_gk = syn_df.groupby(by=metadata)

    raw_flowsize_list = list(raw_gk.size().values)
    syn_flowsize_list = list(syn_gk.size().values)
    metrics_dict["flow_size"] = wasserstein_distance(
        raw_flowsize_list, syn_flowsize_list)

    return metrics_dict


# JSD
def compute_metrics_netflow_v2(raw_df, syn_df):
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
            metric, prstr_raw=True, prstr_syn=False, type="JSD")

    # ts, td, pkt, byt
    for metric in ["ts", "td", "pkt", "byt"]:
        metrics_dict[metric] = jsd(list(raw_df[metric]), list(
            syn_df[metric]), type="continuous")

    return metrics_dict

# JSD + EMD + ranking


def compute_metrics_netflow_v3(raw_df, syn_df):
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

# JSD


def compute_metrics_pcap_v2(raw_df, syn_df):
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
            metric, prstr_raw=True, prstr_syn=False, type="JSD")

    # pkt_len
    for metric in ["pkt_len", "time"]:
        # if metric == "time":
        #     label = "pkt_arrivalTime"
        # else:
        #     label = metric
        metrics_dict[metric] = jsd(list(raw_df[metric]), list(
            syn_df[metric]), type="continuous")

    # interarrival time
    # raw_df = raw_df.sort_values("time")
    # syn_df = syn_df.sort_values("time")
    # metrics_dict["IAT"] = jsd(list(np.diff(raw_df["time"])), list(np.diff(syn_df["time"])), type="continuous")

    # flow size distribution
    metadata = ["srcip", "dstip", "srcport", "dstport", "proto"]
    raw_gk = raw_df.groupby(by=metadata)
    syn_gk = syn_df.groupby(by=metadata)

    raw_flowsize_list = list(raw_gk.size().values)
    syn_flowsize_list = list(syn_gk.size().values)
    metrics_dict["flow_size"] = jsd(
        raw_flowsize_list, syn_flowsize_list, type="continuous")

    return metrics_dict

# JSD + EMD


def compute_metrics_pcap_v3(raw_df, syn_df):
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


def run_netflow_dist_metrics():
    meta_metric_dict_netflow = {}
    for dataset in ["ugr16", "cidds", "ton"]:
        # for dataset in ["ugr16"]:
        # for dataset in ["cidds", "ton"]:
        meta_metric_dict_netflow[dataset] = {}

        raw_df = pd.read_csv(os.path.join(data_baseDir, dataset, "raw.csv"))
        for method in ["CTGAN-A", "CTGAN-B", "STAN", "E-WGAN-GP", "NetShare"]:
            # for method in ["NetShare"]:
            meta_metric_dict_netflow[dataset][method] = {}
            for synid in range(3):
                syn_df_file = os.path.join(
                    data_baseDir,
                    dataset,
                    method,
                    "syn{}.csv".format(synid+1)
                )

                if not os.path.exists(syn_df_file):
                    print("{} not exist.. skipping..".format(syn_df_file))
                    continue
                syn_df = pd.read_csv(syn_df_file)
                print(dataset, method, synid)

                if method == "NetShare":
                    if dataset == "cidds":
                        syn_df["td"] = np.round(syn_df["td"])
                    syn_df["pkt"] = np.round(syn_df["pkt"])
                    syn_df["byt"] = np.round(syn_df["byt"])

                metrics_dict = compute_metrics_netflow_v3(raw_df, syn_df)
                for metric, val in metrics_dict.items():
                    if metric not in meta_metric_dict_netflow[dataset][method]:
                        meta_metric_dict_netflow[dataset][method][metric] = []
                    meta_metric_dict_netflow[dataset][method][metric].append(
                        val)

    print(meta_metric_dict_netflow)

    with open(os.path.join(result_baseDir, "dist_metrics_netflow.json"), 'w') as f:
        json.dump(meta_metric_dict_netflow, f)

    # for dataset, method_metrics in meta_metric_dict_netflow.items():
    #     print(dataset)
    #     for method, metrics_dict in method_metrics.items():
    #         print(method)
    #         for metric in ["srcip", "dstip", "srcport", "dstport", "proto", "td", "pkt", "byt"]:
    #             print("{:.6f}".format(metrics_dict[metric]), end=",")
    #         print("\n")
    #     print("\n")

    return meta_metric_dict_netflow


def run_netflow_qualitative_plots():
    for dataset in ["ugr16", "cidds", "ton"]:
        # for dataset in ["ugr16"]:
        print("Dataset:", dataset)
        # raw_df = rename_netflow(
        #     netflow_csv_file=os.path.join(data_baseDir, dataset, "raw.csv"),
        #     file_type=dataset)
        raw_df = pd.read_csv(os.path.join(data_baseDir, dataset, "raw.csv"))
        result_folder_per_dataset = os.path.join(result_baseDir, dataset)
        os.makedirs(result_folder_per_dataset, exist_ok=True)

        # print(raw_df.head())
        syn_df_dict = {}
        # for method in ["CTGAN-A", "CTGAN-B", "STAN", "E-WGAN-GP", "NetShare"]:
        # for method in ["CTGAN-B", "STAN", "E-WGAN-GP", "NetShare"]:
        for method in ["NetShare"]:
            # for method in ["CTGAN"]:
            # syn_df = rename_netflow(
            #     netflow_csv_file=os.path.join(data_baseDir, dataset, "syn_{}.csv".format(method)),
            #     file_type=dataset
            # )
            syn_df = pd.read_csv(os.path.join(
                data_baseDir, dataset, method, "syn1.csv"))
            syn_df_dict[method] = syn_df
            # print(syn_df.head())

        for metric, xlabel in {
            "srcip": "Source IP popularity rank (log-scale)",
            "dstip": "Dest. IP popularity rank (log-sacle)"
        }.items():
            plot_HH(
                real_df=raw_df,
                syn_df_dict=syn_df_dict,
                xlabel=xlabel,
                ylabel="Relative frequency (log-scale)",
                plot_loc=os.path.join(
                    result_folder_per_dataset,
                    "hh_{}.pdf".format(metric)
                ),
                metric=metric,
                x_logscale=True,
                y_logscale=True
            )

        for metric, xlabel in {
            "srcport": "Top {} service source port number".format(
                N_TOPK_SERVICE_PORTS),
            "dstport": "Top {} service destination port number".format(
                N_TOPK_SERVICE_PORTS),
                "proto": "IP Protocol"}.items():
            print("metric:", metric)
            plot_bar(
                real_df=raw_df,
                # syn_df_dict={k: syn_df_dict[k] for k in ["CTGAN"]},
                syn_df_dict=syn_df_dict,
                xlabel=xlabel,
                ylabel="Relative frequency",
                plot_loc=os.path.join(
                    result_folder_per_dataset,
                    "bar_{}.pdf".format(metric)
                ),
                metric=metric,
                x_logscale=False,
                y_logscale=False,
                data_type="netflow"
            )

        # for metric, xlabel in {"pkt": "# of packets per flow",
        #                         "byt": "# of bytes per flow",
        #                         "td": "Flow duration (seconds)"}.items():
        for metric, xlabel in {
                "flow_size": "# of records with the same five tuple",
                "pkt": "# of packets per flow",
                "byt": "# of bytes per flow",
                "td": "Flow duration (seconds)"}.items():
            plot_cdf(
                raw_df=raw_df,
                syn_df_dict=syn_df_dict,
                xlabel=xlabel,
                ylabel="CDF",
                plot_loc=os.path.join(
                    result_folder_per_dataset,
                    "cdf_{}.pdf".format(metric)),
                metric=metric,
                x_logscale=(metric != "td")
            )


def run_netflow_qualitative_plots_dashboard(
        raw_data_path, syn_data_path, plot_dir):
    raw_df = pd.read_csv(raw_data_path)
    os.makedirs(plot_dir, exist_ok=True)

    syn_df_dict = {}

    for method in ["NetShare"]:
        syn_df = pd.read_csv(syn_data_path)
        syn_df_dict[method] = syn_df

    for metric, xlabel in {
        "srcip": "Source IP popularity rank (log-scale)",
        "dstip": "Dest. IP popularity rank (log-sacle)"
    }.items():
        print("metric:", metric)
        plot_HH(
            real_df=raw_df,
            syn_df_dict=syn_df_dict,
            xlabel=xlabel,
            ylabel="Relative frequency (log-scale)",
            plot_loc=os.path.join(
                plot_dir,
                "hh_{}.png".format(metric)
            ),
            metric=metric,
            x_logscale=True,
            y_logscale=True
        )

    for metric, xlabel in {
        "srcport": "Top {} service source port number".format(
            N_TOPK_SERVICE_PORTS),
        "dstport": "Top {} service destination port number".format(
            N_TOPK_SERVICE_PORTS),
            "proto": "IP Protocol"}.items():
        print("metric:", metric)
        plot_bar(
            real_df=raw_df,
            # syn_df_dict={k: syn_df_dict[k] for k in ["CTGAN"]},
            syn_df_dict=syn_df_dict,
            xlabel=xlabel,
            ylabel="Relative frequency",
            plot_loc=os.path.join(
                plot_dir,
                "bar_{}.png".format(metric)
            ),
            metric=metric,
            x_logscale=False,
            y_logscale=False,
            data_type="netflow"
        )

    # for metric, xlabel in {"pkt": "# of packets per flow",
    #                         "byt": "# of bytes per flow",
    #                         "td": "Flow duration (seconds)"}.items():
    for metric, xlabel in {
            "flow_size": "# of records with the same five tuple",
            "pkt": "# of packets per flow",
            "byt": "# of bytes per flow",
            "td": "Flow duration (seconds)"}.items():
        print("metric:", metric)
        plot_cdf(
            raw_df=raw_df,
            syn_df_dict=syn_df_dict,
            xlabel=xlabel,
            ylabel="CDF",
            plot_loc=os.path.join(
                plot_dir,
                "cdf_{}.png".format(metric)),
            metric=metric,
            x_logscale=(metric != "td")
        )


def run_pcap_dist_metrics():
    meta_metric_dict_pcap = {}
    for dataset in ["caida", "dc", "ca"]:
        # for dataset in ["caida", "dc"]:
        # for dataset in ["caida", "dc"]:
        # for dataset in ["ca"]:
        meta_metric_dict_pcap[dataset] = {}

        raw_df = pd.read_csv(os.path.join(data_baseDir, dataset, "raw.csv"))

        for method in [
            "CTGAN-A", "CTGAN-B", "PAC-GAN", "PacketCGAN", "Flow-WGAN",
                "NetShare"]:
            meta_metric_dict_pcap[dataset][method] = {}
            for synid in range(3):
                syn_df_file = os.path.join(
                    data_baseDir,
                    dataset,
                    method,
                    "syn{}.csv".format(synid+1)
                )

                if not os.path.exists(syn_df_file):
                    print("{} not exist.. skipping..".format(syn_df_file))
                    continue
                syn_df = pd.read_csv(syn_df_file)
                print(dataset, method, synid)

                metrics_dict = compute_metrics_pcap_v3(raw_df, syn_df)
                for metric, val in metrics_dict.items():
                    if metric == "PIAT":  # exclude
                        continue
                    if metric not in meta_metric_dict_pcap[dataset][method]:
                        meta_metric_dict_pcap[dataset][method][metric] = []
                    meta_metric_dict_pcap[dataset][method][metric].append(val)

    with open(os.path.join(result_baseDir, "dist_metrics_pcap.json"), 'w') as f:
        json.dump(meta_metric_dict_pcap, f)

    return meta_metric_dict_pcap


def run_pcap_qualitative_plots():
    # for dataset in ["caida", "data_center", "cyber_attack"]:
    for dataset in ["caida", "dc", "ca"]:
        # for dataset in ["caida"]:
        print("Dataset:", dataset)
        result_folder_per_dataset = os.path.join(result_baseDir, dataset)
        os.makedirs(result_folder_per_dataset, exist_ok=True)

        raw_df = pd.read_csv(os.path.join(data_baseDir, dataset, "raw.csv"))

        # print(raw_df.head())
        syn_df_dict = {}
        # for method in ["PAC-GAN", "PacketCGAN", "Flow-WGAN", "NetShare"]:
        for method in ["NetShare"]:
            # for method in ["CTGAN-B", "PAC-GAN", "PacketCGAN", "Flow-WGAN", "NetShare"]:
            # if not os.path.exists(os.path.join(data_baseDir, dataset, "syn_{}.csv".format(method))):
            #     continue

            syn_df = pd.read_csv(os.path.join(
                data_baseDir, dataset, method, "syn1.csv".format(method)))
            syn_df_dict[method] = syn_df
            # print(syn_df.head())

        # print(syn_df_dict)

        for metric, xlabel in {
            "srcip": "Source IP popularity rank (log-scale)",
            "dstip": "Dest. IP popularity rank (log-sacle)"
        }.items():
            plot_HH(
                real_df=raw_df,
                syn_df_dict=syn_df_dict,
                xlabel=xlabel,
                ylabel="Relative frequency (log-scale)",
                plot_loc=os.path.join(
                    result_folder_per_dataset,
                    "hh_{}.pdf".format(metric)
                ),
                metric=metric,
                x_logscale=True,
                y_logscale=True
            )

        for metric, xlabel in {
            "srcport": "Top {} service source port number".format(
                N_TOPK_SERVICE_PORTS),
            "dstport": "Top {} service destination port number".format(
                N_TOPK_SERVICE_PORTS),
                "proto": "IP Protocol"}.items():
            print("metric:", metric)
            plot_bar(
                real_df=raw_df,
                # syn_df_dict={k: syn_df_dict[k] for k in ["CTGAN"]},
                syn_df_dict=syn_df_dict,
                xlabel=xlabel,
                ylabel="Relative frequency",
                plot_loc=os.path.join(
                    result_folder_per_dataset,
                    "bar_{}.pdf".format(metric)
                ),
                metric=metric,
                x_logscale=False,
                y_logscale=False,
                data_type="pcap"
            )

        for metric, xlabel in {
            "pkt_len": "Packet size (bytes)",
            "flow_size": "Flow size (# of packets perflow)"
        }.items():
            plot_cdf(
                raw_df=raw_df,
                syn_df_dict=syn_df_dict,
                xlabel=xlabel,
                ylabel="CDF",
                plot_loc=os.path.join(
                    result_folder_per_dataset,
                    "cdf_{}.pdf".format(metric)),
                metric=metric,
                x_logscale=(metric != "td")
            )

# raw_data_path: csv format


def run_pcap_qualitative_plots_dashboard(
        raw_data_path, syn_data_path, plot_dir):
    raw_df = pd.read_csv(raw_data_path)
    os.makedirs(plot_dir, exist_ok=True)

    # print(raw_df.head())
    syn_df_dict = {}
    # for method in ["PAC-GAN", "PacketCGAN", "Flow-WGAN", "NetShare"]:
    for method in ["NetShare"]:
        syn_df = pd.read_csv(syn_data_path)
        syn_df_dict[method] = syn_df

    for metric, xlabel in {
        "srcip": "Source IP popularity rank (log-scale)",
        "dstip": "Dest. IP popularity rank (log-sacle)"
    }.items():
        plot_HH(
            real_df=raw_df,
            syn_df_dict=syn_df_dict,
            xlabel=xlabel,
            ylabel="Relative frequency (log-scale)",
            plot_loc=os.path.join(
                plot_dir,
                "hh_{}.png".format(metric)
            ),
            metric=metric,
            x_logscale=True,
            y_logscale=True
        )

    for metric, xlabel in {
        "srcport": "Top {} service source port number".format(
            N_TOPK_SERVICE_PORTS),
        "dstport": "Top {} service destination port number".format(
            N_TOPK_SERVICE_PORTS),
            "proto": "IP Protocol"}.items():
        print("metric:", metric)
        plot_bar(
            real_df=raw_df,
            # syn_df_dict={k: syn_df_dict[k] for k in ["CTGAN"]},
            syn_df_dict=syn_df_dict,
            xlabel=xlabel,
            ylabel="Relative frequency",
            plot_loc=os.path.join(
                plot_dir,
                "bar_{}.png".format(metric)
            ),
            metric=metric,
            x_logscale=False,
            y_logscale=False,
            data_type="pcap"
        )

    for metric, xlabel in {
        "pkt_len": "Packet size (bytes)",
        "flow_size": "Flow size (# of packets perflow)"
    }.items():
        plot_cdf(
            raw_df=raw_df,
            syn_df_dict=syn_df_dict,
            xlabel=xlabel,
            ylabel="CDF",
            plot_loc=os.path.join(
                plot_dir,
                "cdf_{}.png".format(metric)),
            metric=metric,
            x_logscale=(metric != "td")
        )


def run_netflow_dist_metrics_privacy():
    for dataset in ["ugr16"]:
        raw_df = pd.read_csv("../data/1M/{}/raw.csv".format(dataset))
        for filename in [
            "syn_df_dpnaive.csv", "syn_df_dp_public_diff.csv",
                "syn_df_dp_public_same.csv"]:
            fidelity_list = []

            # for dp_multi in [0.01, 0.1, 0.5, 1.0, 2.0, 3.0, 4.0]:
            for dp_multi in [4.0, 3.0, 2.0, 1.0, 0.5, 0.1, 0.01]:
                syn_df_file = os.path.join(
                    "../results/results_sigcomm2022/raw_privacy/",
                    "{},dp_noise_multiplier-{}".format(dataset, dp_multi),
                    filename
                )

                syn_df = pd.read_csv(syn_df_file)

                metrics_dict = compute_metrics_netflow_v2(raw_df, syn_df)

                print(dataset, dp_multi, np.average(
                    list(metrics_dict.values())))

                fidelity_list.append(np.average(list(metrics_dict.values())))

            print(filename, fidelity_list)


def run_pcap_dist_metrics_privacy():
    for dataset in ["caida"]:
        raw_df = pd.read_csv("../data/1M/{}/raw.csv".format(dataset))
        for filename in [
            "syn_df_dpnaive.csv", "syn_df_dp_public_diff.csv",
                "syn_df_dp_public_same.csv"]:
            fidelity_list = []

            # for dp_multi in [0.01, 0.1, 0.5, 1.0, 2.0, 3.0, 4.0]:
            for dp_multi in [4.0, 3.0, 2.0, 1.0, 0.5, 0.1, 0.01]:
                syn_df_file = os.path.join(
                    "../results/results_sigcomm2022/raw_privacy/",
                    "{},dp_noise_multiplier-{}".format(dataset, dp_multi),
                    filename
                )

                syn_df = pd.read_csv(syn_df_file)

                metrics_dict = compute_metrics_pcap_v2(raw_df, syn_df)

                print(dataset, dp_multi, np.average(
                    list(metrics_dict.values())))

                fidelity_list.append(np.average(list(metrics_dict.values())))

            print(filename, fidelity_list)


def run_pcap_dist_metrics_SN():
    dict_scale_fiedlity = {}

    for dataset in ["caida"]:
        raw_df = pd.read_csv("../data/1M/{}/raw.csv".format(dataset))

        for scale in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]:
            syn_df_file = os.path.join(
                "../results/results_sigcomm2022/SN/",
                "{},sn_scale-{},interarrival-False".format(dataset, scale),
                "syn_df.csv"
            )

            syn_df = pd.read_csv(syn_df_file)

            metrics_dict = compute_metrics_pcap_v2(raw_df, syn_df)

            print("scale", scale, np.average(list(metrics_dict.values())))

            dict_scale_fiedlity[scale] = np.average(
                list(metrics_dict.values()))

    print(dict_scale_fiedlity)


def run_ton_dstport_plot():
    for dataset in ["ton"]:
        print("Dataset:", dataset)
        raw_df = pd.read_csv(os.path.join(data_baseDir, dataset, "raw.csv"))
        result_folder_per_dataset = os.path.join(result_baseDir, dataset)
        os.makedirs(result_folder_per_dataset, exist_ok=True)

        syn_df_dict = {}
        for method in ["CTGAN-B", "STAN", "E-WGAN-GP", "NetShare"]:
            syn_df = pd.read_csv(os.path.join(
                data_baseDir, dataset, method, "syn1.csv"))
            syn_df_dict[method] = syn_df
            # print(syn_df.head())

        for metric, xlabel in {
            # "srcport": "Top {} service source port number".format(N_TOPK_SERVICE_PORTS),
            "dstport": "Top {} service destination port number".format(N_TOPK_SERVICE_PORTS),
            # "proto": "IP Protocol"
        }.items():
            print("metric:", metric)
            plot_bar_port(
                real_df=raw_df,
                syn_df_dict=syn_df_dict,
                xlabel=xlabel,
                ylabel="Relative frequency",
                plot_loc=os.path.join(
                    result_folder_per_dataset,
                    "bar_{}.pdf".format(metric)
                ),
                metric=metric,
                x_logscale=False,
                y_logscale=False,
                data_type="netflow"
            )


def run_ugr16_flowsize_pkt_byt():
    for dataset in ["ugr16"]:
        print("Dataset:", dataset)
        raw_df = pd.read_csv(os.path.join(data_baseDir, dataset, "raw.csv"))
        result_folder_per_dataset = os.path.join(result_baseDir, dataset)
        os.makedirs(result_folder_per_dataset, exist_ok=True)

        syn_df_dict = {}
        for method in ["CTGAN-B", "STAN", "E-WGAN-GP", "NetShare"]:
            syn_df = pd.read_csv(os.path.join(
                data_baseDir, dataset, method, "syn1.csv"))
            syn_df_dict[method] = syn_df

        for metric, xlabel in {
            "flow_size": "# of records with the same five tuple",
            "pkt": "# of packets per flow",
            "byt": "# of bytes per flow",
            # "td": "Flow duration (seconds)",
        }.items():
            plot_cdf(
                raw_df=raw_df,
                syn_df_dict=syn_df_dict,
                xlabel=xlabel,
                ylabel="CDF",
                plot_loc=os.path.join(
                    result_folder_per_dataset,
                    "cdf_{}.pdf".format(metric)),
                metric=metric,
                x_logscale=(metric != "td")
            )


def run_caida_flowsize():
    for dataset in ["caida"]:
        print("Dataset:", dataset)
        result_folder_per_dataset = os.path.join(result_baseDir, dataset)
        os.makedirs(result_folder_per_dataset, exist_ok=True)

        raw_df = pd.read_csv(os.path.join(data_baseDir, dataset, "raw.csv"))

        syn_df_dict = {}
        for method in [
                "CTGAN-B", "PAC-GAN", "PacketCGAN", "Flow-WGAN", "NetShare"]:
            syn_df = pd.read_csv(os.path.join(
                data_baseDir, dataset, method, "syn1.csv".format(method)))
            syn_df_dict[method] = syn_df

        for metric, xlabel in {
            "flow_size": "Flow size (# of packets perflow)"
        }.items():
            plot_cdf(
                raw_df=raw_df,
                syn_df_dict=syn_df_dict,
                xlabel=xlabel,
                ylabel="CDF",
                plot_loc=os.path.join(
                    result_folder_per_dataset,
                    "cdf_{}.pdf".format(metric)),
                metric=metric,
                x_logscale=(metric != "td")
            )


if __name__ == '__main__':
    # run_netflow_dist_metrics()
    run_netflow_qualitative_plots()
    # run_pcap_dist_metrics()
    run_pcap_qualitative_plots()

    # run_netflow_dist_metrics_privacy()

    # run_pcap_dist_metrics_privacy()

    # run_pcap_dist_metrics_SN()

    # PLOTTING DISTRIBUTIONAL METRICS (EMD + JSD)
    # run_netflow_dist_metrics()
    # run_pcap_dist_metrics()

    # VISUAL PLOTS FOR PAPER
    # run_ton_dstport_plot()
    # run_ugr16_flowsize_pkt_byt()
    # run_caida_flowsize()
