import os, sys, copy, json, importlib, random
import numpy as np
import pandas as pd
import socket, struct, ipaddress
import argparse

from tqdm import tqdm
from scapy.all import IP, ICMP, TCP, UDP
from scapy.all import wrpcap
from scipy.stats import rankdata

sys.path.append("../eval")
from dist_metrics import compute_metrics_netflow_v3, compute_metrics_pcap_v3

from gan.util import configs2configsgroup, load_config

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

# chunk_folder: "chunk_id-0"
def get_per_chunk_df(chunk_folder):
    df_names = [file for file in os.listdir(chunk_folder) if file.endswith(".csv")]
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

def main(args):
    config = getattr(importlib.import_module("configs."+args.config_file), 'config')
    configs = load_config(config)
    random.seed(42)
    random.shuffle(configs)
    configs, config_group_list = configs2configsgroup(configs, generation_flag=True)

    print("\n\n")
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
        assert len([file for file in os.listdir(syndf_root_folder) if file.startswith("chunk_id")]) == len(config_ids)

        best_syndfs = []
        for chunk_id, config_idx in enumerate(config_ids):
            config = configs[config_idx]
            raw_df = pd.read_csv(os.path.join(
                "../data", 
                config["dataset"],
                "raw.csv"
                ))
            
            if "/caida/" in config["dataset"] or "/dc/" in config["dataset"] or "/ca/" in config["dataset"]:
                data_type = "pcap"
                time_col_name = "time"
            elif "ugr16" in config["dataset"] or "cidds" in config["dataset"] or "ton" in config["dataset"]:
                data_type = "netflow"
                time_col_name = "ts"
            else:
                raise ValueError("Unknown data type! Must be netflow or pcap...")
            
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
                    if args.truncate == "per_chunk":
                        syn_df = syn_df[(syn_df[time_col_name] >= raw_df[time_col_name].min()) & (syn_df[time_col_name] <= raw_df[time_col_name].max())]
                    elif args.truncate == "none":
                        pass
                    # TODO: support more truncation methods if necessary
                    else:
                        raise ValueError("Unknown truncation methods...")

                    syn_dfs.append(syn_df)
            
            best_syndf_idx, best_syndf = compare_rawdf_syndfs(raw_df, syn_dfs, data_type)
            best_syndfs.append(best_syndf)
            
            print("Chunk_id: {}, # of syn dfs: {}, best_syndf: {}".format(chunk_id, len(syn_dfs), syn_dfs_names[best_syndf_idx]))
        
        big_best_syndf = pd.concat(best_syndfs)
        print("Big syndf shape:", big_best_syndf.shape)
        print()

        if (config_group["dataset"], config_group["dp_noise_multiplier"]) not in dict_dataset_syndfs:
            dict_dataset_syndfs[(config_group["dataset"], config_group["dp_noise_multiplier"])] = []
        dict_dataset_syndfs[(config_group["dataset"], config_group["dp_noise_multiplier"])].append(big_best_syndf)
    
    dict_dataset_bestsyndf = {}
    for dataset_dpnoisemultiplier_pair, syn_dfs in dict_dataset_syndfs.items():
        assert len(syn_dfs) >= 1

        if len(syn_dfs) > 1:
            raw_df = pd.read_csv(os.path.join("../data", dataset_dpnoisemultiplier_pair[0], "raw.csv"))
            dataset = dataset_dpnoisemultiplier_pair[0]
            if "caida" in dataset or "dc" in dataset or "ca" in dataset:
                data_type = "pcap"
            elif "ugr16" in dataset or "cidds" in dataset or "ton" in dataset:
                data_type = "netflow"
            else:
                raise ValueError("Unknown data type! Must be netflow or pcap...")

            best_syndf_idx, best_syn_df = compare_rawdf_syndfs(raw_df, syn_dfs, data_type)
            dict_dataset_bestsyndf[dataset_dpnoisemultiplier_pair] = best_syn_df
        else:
            dict_dataset_bestsyndf[dataset_dpnoisemultiplier_pair] = syn_dfs[0]

    print("Aggregated final dataset syndf")
    for dataset_dpnoisemultiplier_pair, best_syndf in dict_dataset_bestsyndf.items():
        dataset = dataset_dpnoisemultiplier_pair[0]
        dp_noise_multiplier = dataset_dpnoisemultiplier_pair[1]
        print(dataset_dpnoisemultiplier_pair, best_syndf.shape)
        best_syndf_folder = os.path.join(
            configs[0]["result_root_folder"], # the single config file uses the same result_root_folder by default
            dataset
        )
        os.makedirs(best_syndf_folder, exist_ok=True)

        # find best syndf index i.e., for evaluation fairness
        cur_max_idx = None
        for file in os.listdir(best_syndf_folder):
            if file.startswith("syn_df,dp_noise_multiplier-{},truncate-{},id-".format(dp_noise_multiplier, args.truncate)):
                this_id = int(os.path.splitext(file)[0].split(",")[-1].split("-")[1])
                if cur_max_idx == None or this_id > cur_max_idx:
                    cur_max_idx = this_id
        if cur_max_idx == None:
            cur_max_idx = 1
        else:
            cur_max_idx += 1

        best_syndf_filename = os.path.join(
            best_syndf_folder,
            "syn_df,dp_noise_multiplier-{},truncate-{},id-{}.csv".format(dp_noise_multiplier, args.truncate, cur_max_idx)
        )

        print("best_syn_df filename:", best_syndf_filename)

        # sort by timestamp
        if "caida" in dataset or "dc" in dataset or "ca" in dataset:
            time_col_name = "time"
        elif "ugr16" in dataset or "cidds" in dataset or "ton" in dataset:
            time_col_name = "ts"
        else:
            raise ValueError("Unknown data type! Must be netflow or pcap...")
        
        best_syndf = best_syndf.sort_values(time_col_name)
        best_syndf.to_csv(best_syndf_filename, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_file', type=str)
    # truncate:
    #   per_chunk
    #   none
    parser.add_argument('--truncate', type=str, default="per_chunk")

    args = parser.parse_args()
    main(args)
