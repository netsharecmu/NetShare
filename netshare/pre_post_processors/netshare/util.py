import os
import pickle
import math
import socket
import struct
import ipaddress
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from scapy.all import IP, ICMP, TCP, UDP
from scapy.all import wrpcap
from scipy.stats import rankdata
from .embedding_helper import (
    build_annoy_dictionary_word2vec,
    get_original_obj
)
from .dist_metrics import (
    compute_metrics_netflow_v3,
    compute_metrics_pcap_v3
)


def denormalize(data_attribute, data_feature, data_gen_flag, config):
    df_list = []

    interarrival_flag = True if config["timestamp"] == "interarrival" else False
    file_type = config["dataset_type"]
    WORD2VEC_SIZE = config["word2vec_vecSize"]
    ENCODE_IP = "bit"
    raw_data_folder = config["dataset"]

    fin = open(os.path.join(raw_data_folder, "fields.pkl"), "rb")
    fields = pickle.load(fin)
    fin.close()

    word2vec_model_path = os.path.join(_last_lvl_folder(
        raw_data_folder), "word2vec_vecSize_{}.model".format(WORD2VEC_SIZE))

    per_chunk_raw_df = pd.read_csv(os.path.join(raw_data_folder, "raw.csv"))
    ann_port, port_dic, ann_proto, proto_dic = build_annoy_dictionary_word2vec(
        csv=per_chunk_raw_df,
        model=word2vec_model_path,
        length=WORD2VEC_SIZE,
        file_type=file_type,
        n_trees=1000,
        encode_IP=ENCODE_IP)

    print("Finish building annoy dictionary")

    attr_set = set()

    for i in tqdm(range(np.shape(data_attribute)[0])):
        attr_per_row = data_attribute[i]

        srcip = list(attr_per_row[0:64])
        dstip = list(attr_per_row[64:128])
        srcport = list(attr_per_row[128:128 + WORD2VEC_SIZE])
        dstport = list(
            attr_per_row[128 + WORD2VEC_SIZE:128 + WORD2VEC_SIZE * 2])
        proto = list(
            attr_per_row[128 + WORD2VEC_SIZE * 2:128 + WORD2VEC_SIZE * 3])

        srcip = fields["srcip"].denormalize(srcip)
        dstip = fields["dstip"].denormalize(dstip)
        srcport = get_original_obj(ann_port, srcport, port_dic)
        dstport = get_original_obj(ann_port, dstport, port_dic)
        proto = get_original_obj(ann_proto, proto, proto_dic)

        # use flow start + interarrival
        if interarrival_flag == True:
            flow_start_time = fields["flow_start"].denormalize(
                attr_per_row[128 + WORD2VEC_SIZE * 3])
            cur_pkt_time = flow_start_time

        # remove duplicated attributes (five tuples)
        if (srcip, dstip, srcport, dstport, proto) in attr_set:
            continue
        attr_set.add((srcip, dstip, srcport, dstport, proto))

        for j in range(np.shape(data_feature)[1]):
            if data_gen_flag[i][j] == 1.0:
                df_per_row = {}

                df_per_row["srcip"] = srcip
                df_per_row["dstip"] = dstip
                df_per_row["srcport"] = srcport
                df_per_row["dstport"] = dstport
                df_per_row["proto"] = proto

                if file_type == "netflow":
                    time_col = "ts"
                elif file_type == "pcap":
                    time_col = "time"
                else:
                    raise ValueError(
                        "Unknown file type! Currently support PCAP and NetFlow...")
                if interarrival_flag == False:
                    df_per_row[time_col] = fields[time_col].denormalize(
                        data_feature[i][j][0])
                else:
                    if j == 0:
                        df_per_row[time_col] = flow_start_time
                    else:
                        df_per_row[time_col] = cur_pkt_time + \
                            fields["interarrival_within_flow"].denormalize(
                                data_feature[i][j][0])
                        cur_pkt_time = df_per_row[time_col]

                if file_type == "netflow":
                    df_per_row["td"] = math.exp(
                        fields["td"].denormalize(data_feature[i][j][1])) - 1
                    df_per_row["pkt"] = math.exp(
                        fields["pkt"].denormalize(data_feature[i][j][2])) - 1
                    df_per_row["byt"] = math.exp(
                        fields["byt"].denormalize(data_feature[i][j][3])) - 1

                    cur_idx = 4
                    for field in ['label', 'type']:
                        if field in fields:
                            field_len = len(fields[field].choices)
                            df_per_row[field] = fields[field].denormalize(
                                list(data_feature[i][j]
                                     [cur_idx:(cur_idx + field_len)])
                            )
                            cur_idx += field_len

                elif file_type == "pcap":
                    df_per_row["pkt_len"] = fields["pkt_len"].denormalize(
                        data_feature[i][j][1])
                    df_per_row["tos"] = fields["tos"].denormalize(
                        data_feature[i][j][2])
                    df_per_row["id"] = fields["id"].denormalize(
                        data_feature[i][j][3])
                    df_per_row["flag"] = fields["flag"].denormalize(
                        list(data_feature[i][j][4:7]))
                    df_per_row["off"] = fields["off"].denormalize(
                        data_feature[i][j][7])
                    df_per_row["ttl"] = fields["ttl"].denormalize(
                        data_feature[i][j][8])

                else:
                    raise ValueError(
                        "Unknown file type! Currently support PCAP and NetFlow...")

                df_list.append(df_per_row)

    df = pd.DataFrame(df_list)
    print(df.head())

    return df


def _last_lvl_folder(folder):
    return str(Path(folder).parents[0])


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
        except BaseException:
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
            if file.startswith("syn_df,dp_noise_multiplier-{},truncate-{},id-".format(
                    dp_noise_multiplier, config["truncate"])):
                this_id = int(os.path.splitext(file)[
                              0].split(",")[-1].split("-")[1])
                if cur_max_idx is None or this_id > cur_max_idx:
                    cur_max_idx = this_id
        if cur_max_idx is None:
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
