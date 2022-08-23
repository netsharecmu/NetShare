import os
import math
import ipaddress

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import netshare.ray as ray

from tqdm import tqdm
from gensim.models import Word2Vec, word2vec
from sklearn import preprocessing

from netshare.utils import Normalization
from netshare.utils import DiscreteField, ContinuousField, BitField
from .embedding_helper import get_vector


def countList2cdf(count_list):
    # dist_dict: {key : count}
    dist_dict = {}
    for x in count_list:
        if x not in dist_dict:
            dist_dict[x] = 0
        dist_dict[x] += 1
    dist_dict = {k: v for k, v in sorted(
        dist_dict.items(), key=lambda x: x[0])}
    x = dist_dict.keys()

    pdf = np.asarray(list(dist_dict.values()), dtype=float) / \
        float(sum(dist_dict.values()))
    cdf = np.cumsum(pdf)

    return x, cdf


def plot_cdf(count_list, xlabel, ylabel, title, filename, base_dir):
    x, cdf = countList2cdf(count_list)
    plt.plot(x, cdf, linewidth=2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(os.path.join(base_dir, filename), bbox_inches="tight", dpi=300)
    plt.close()


def continuous_list_flag(l_):
    '''
    # l: [1, 2, 3, 4]: True
    # [1, 3, 5, 7]: False
    '''
    first_order_diff = np.diff(l_)
    return len(set(first_order_diff)) <= 1


def chunks(a, n):
    '''Split list *a* into *n* chunks evenly'''
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def divide_chunks(l_, n):
    '''Yield successive n-sized chunks from l.'''
    # looping till length l
    for i in range(0, len(l_), n):
        yield l_[i:i + n]


def IP_int2str(IP_int):
    return str(ipaddress.ip_address(IP_int))


def IP_str2int(IP_str):
    return int(ipaddress.ip_address(IP_str))


def IPs_int2str(IPs_int):
    return [IP_int2str(i) for i in IPs_int]


def IPs_str2int(IPs_str):
    return [IP_str2int(i) for i in IPs_str]


def df2chunks(big_raw_df, file_type,
              split_type="fixed_size", n_chunks=10, eps=1e-5):
    if file_type == "pcap":
        time_col_name = "time"
    elif file_type == "netflow":
        time_col_name = "ts"

    # sanity sort
    big_raw_df = big_raw_df.sort_values(time_col_name)

    dfs = []
    if split_type == "fixed_size":
        chunk_size = math.ceil(big_raw_df.shape[0] / n_chunks)
        for chunk_id in range(n_chunks):
            df_chunk = big_raw_df.iloc[chunk_id *
                                       chunk_size:((chunk_id+1)*chunk_size)]
            dfs.append(df_chunk)
        return dfs, chunk_size

    elif split_type == "fixed_time":
        time_evenly_spaced = np.linspace(big_raw_df[time_col_name].min(
        ), big_raw_df[time_col_name].max(), num=n_chunks+1)
        time_evenly_spaced[-1] *= (1+eps)

        chunk_time = (big_raw_df[time_col_name].max() -
                      big_raw_df[time_col_name].min()) / n_chunks

        for chunk_id in range(n_chunks):
            df_chunk = big_raw_df[
                (big_raw_df[time_col_name] >= time_evenly_spaced[chunk_id]) &
                (big_raw_df[time_col_name] < time_evenly_spaced[chunk_id+1])]
            if len(df_chunk) == 0:
                print("Raw chunk_id: {}, empty df_chunk!".format(chunk_id))
                continue
            dfs.append(df_chunk)
        return dfs, chunk_time

    else:
        raise ValueError("Unknown split type")


@ray.remote(scheduling_strategy="SPREAD", max_calls=1)
def split_per_chunk(
        config,
        fields,
        df_per_chunk,
        embed_model,
        global_max_flow_len,
        chunk_id,
        flowkeys_chunkidx=None,
):
    if config["dataset_type"] == "pcap":
        time_col = "time"
    elif config["dataset_type"] == "netflow":
        time_col = "ts"
    split_name = config["split_name"]
    word2vec_vecSize = config["word2vec_vecSize"]
    file_type = config["dataset_type"]
    encode_IP = config["encode_IP"]
    num_chunks = config["n_chunks"]

    # w/o DP: normalize time by per-chunk min/max
    if config["timestamp"] == "interarrival":
        gk = df_per_chunk.groupby(
            ["srcip", "dstip", "srcport", "dstport", "proto"])
        flow_start_list = []
        interarrival_within_flow_list = []
        for group_name, df_group in gk:
            flow_start_list.append(df_group.iloc[0][time_col])
            interarrival_within_flow_list += [0.0] + \
                list(np.diff(df_group[time_col]))
        fields["flow_start"].min_x = float(min(flow_start_list))
        fields["flow_start"].max_x = float(max(flow_start_list))
        fields["interarrival_within_flow"].min_x = float(
            min(interarrival_within_flow_list))
        fields["interarrival_within_flow"].max_x = float(
            max(interarrival_within_flow_list))
    elif config["timestamp"] == "raw":
        fields["ts"].min_x = float(df_per_chunk[time_col].min())
        fields["ts"].max_x = float(df_per_chunk[time_col].max())

    if "multichunk_dep" in split_name and flowkeys_chunkidx is None:
        raise ValueError(
            "Cross-chunk mechanism enabled, \
                cross-chunk flow stats not provided!")

    metadata = ["srcip", "dstip", "srcport", "dstport", "proto"]
    gk = df_per_chunk.groupby(by=metadata)

    data_attribute = []
    data_feature = []
    data_gen_flag = []
    num_flows_startFromThisChunk = 0

    for group_name, df_group in tqdm(gk):
        # RESET INDEX TO MAKE IT START FROM ZERO
        df_group = df_group.reset_index(drop=True)

        attr_per_row = []
        feature_per_row = []
        data_gen_flag_per_row = []

        # metadata
        # word2vec
        if encode_IP == 'word2vec':
            attr_per_row += list(get_vector(embed_model,
                                 str(group_name[0]), norm_option=True))
            attr_per_row += list(get_vector(embed_model,
                                 str(group_name[1]), norm_option=True))
        # bitwise
        elif encode_IP == 'bit':
            attr_per_row += fields["srcip"].normalize(group_name[0])
            attr_per_row += fields["dstip"].normalize(group_name[1])

        attr_per_row += list(get_vector(embed_model,
                             str(group_name[2]), norm_option=True))
        attr_per_row += list(get_vector(embed_model,
                             str(group_name[3]), norm_option=True))
        attr_per_row += list(get_vector(embed_model,
                             str(group_name[4]), norm_option=True))
        attr_per_row.append(
            fields["flow_start"].normalize(df_group.iloc[0][time_col]))

        # cross-chunk generation
        if "multichunk_dep" in split_name:
            if str(group_name) in flowkeys_chunkidx:  # sanity check
                # flow starts from this chunk
                if flowkeys_chunkidx[str(group_name)][0] == chunk_id:
                    attr_per_row += fields["startFromThisChunk"].normalize(1.0)
                    num_flows_startFromThisChunk += 1

                    for i in range(num_chunks):
                        if i in flowkeys_chunkidx[str(group_name)]:
                            attr_per_row += fields["chunk_{}".format(
                                i)].normalize(1.0)
                        else:
                            attr_per_row += fields["chunk_{}".format(
                                i)].normalize(0.0)

                # flow does not start from this chunk
                else:
                    attr_per_row += fields["startFromThisChunk"].normalize(0.0)
                    if split_name == "multichunk_dep_v1":
                        for i in range(num_chunks):
                            attr_per_row += fields["chunk_{}".format(
                                i)].normalize(0.0)

                    elif split_name == "multichunk_dep_v2":
                        for i in range(num_chunks):
                            if i in flowkeys_chunkidx[str(group_name)]:
                                attr_per_row += fields["chunk_{}".format(
                                    i)].normalize(1.0)
                            else:
                                attr_per_row += fields["chunk_{}".format(
                                    i)].normalize(0.0)

        data_attribute.append(attr_per_row)

        # measurement
        interarrival_per_flow_list = [0.0] + list(np.diff(df_group[time_col]))
        for row_index, row in df_group.iterrows():
            timeseries_per_step = []

            # timestamp: raw/interarrival
            if config["timestamp"] == "interarrival":
                timeseries_per_step.append(
                    fields["interarrival_within_flow"].normalize
                    (interarrival_per_flow_list[row_index]))
            elif config["timestamp"] == "raw":
                timeseries_per_step.append(
                    fields[time_col].normalize(row[time_col]))

            if file_type == "pcap":
                timeseries_per_step.append(row["pkt_len"])
                # full IP header
                if config["full_IP_header"]:
                    for field in ["tos", "id", "flag", "off", "ttl"]:
                        if isinstance(fields[field], DiscreteField):
                            timeseries_per_step += \
                                fields[field].normalize(row[field])
                        else:
                            timeseries_per_step.append(row[field])
            elif file_type == "netflow":
                timeseries_per_step += [row["td"], row["pkt"], row["byt"]]
                for field in ['label', 'type']:
                    if field in df_per_chunk.columns:
                        timeseries_per_step += fields[field].normalize(
                            row[field])

            feature_per_row.append(timeseries_per_step)
            data_gen_flag_per_row.append(1.0)

        data_feature.append(feature_per_row)
        data_gen_flag.append(data_gen_flag_per_row)

    data_attribute = np.asarray(data_attribute)
    data_feature = np.asarray(data_feature)
    data_gen_flag = np.asarray(data_gen_flag)

    print("data_attribute: {}, {}GB in memory".format(
        np.shape(data_attribute),
        data_attribute.size*data_attribute.itemsize/1024/1024/1024))
    print("data_feature: {}, {}GB in memory".format(
        np.shape(data_feature),
        data_feature.size*data_feature.itemsize/1024/1024/1024))
    print("data_gen_flag: {}, {}GB in memory".format(
        np.shape(data_gen_flag),
        data_gen_flag.size*data_gen_flag.itemsize/1024/1024/1024))

    data_attribute_output = []
    data_feature_output = []

    if encode_IP == 'word2vec':
        for flow_key in ["srcip", "dstip"]:
            for i in range(word2vec_vecSize):
                data_attribute_output.append(
                    fields["{}_{}".format(flow_key, i)].getOutputType())
    elif encode_IP == 'bit':
        data_attribute_output += fields["srcip"].getOutputType()
        data_attribute_output += fields["dstip"].getOutputType()
    for flow_key in ["srcport", "dstport", "proto"]:
        for i in range(word2vec_vecSize):
            data_attribute_output.append(
                fields["{}_{}".format(flow_key, i)].getOutputType())

    if config["timestamp"] == "interarrival":
        data_attribute_output.append(fields["flow_start"].getOutputType())

    if "multichunk_dep" in split_name:
        data_attribute_output.append(
            fields["startFromThisChunk"].getOutputType())
        for i in range(num_chunks):
            data_attribute_output.append(
                fields["chunk_{}".format(i)].getOutputType())

    if config["timestamp"] == "interarrival":
        field_list = ["interarrival_within_flow"]
    elif config["timestamp"] == "raw":
        field_list = [time_col]

    if file_type == "pcap":
        if config["full_IP_header"]:
            field_list += ["pkt_len", "tos", "id", "flag", "off", "ttl"]
        else:
            field_list += ["pkt_len"]

    elif file_type == "netflow":
        field_list += ["td", "pkt", "byt"]
        for field in ['label', 'type']:
            if field in df_per_chunk.columns:
                field_list.append(field)

    for field in field_list:
        field_output = fields[field].getOutputType()
        if isinstance(field_output, list):
            data_feature_output += field_output
        else:
            data_feature_output.append(field_output)

    print("data_attribute_output:", len(data_attribute_output))
    print("data_feature_output:", len(data_feature_output))

    return data_attribute, data_feature, data_gen_flag, \
        data_attribute_output, data_feature_output, fields
