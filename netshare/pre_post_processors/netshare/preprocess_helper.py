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


def df2chunks(big_raw_df,
              config_timestamp,
              split_type="fixed_size",
              n_chunks=10,
              eps=1e-5):

    if n_chunks == 1:
        return [big_raw_df]

    if n_chunks > 1 and \
            ((not config_timestamp["column"]) or (not config_timestamp["generation"])):
        raise ValueError(
            "Trying to split into multiple chunks by timestamp but no timestamp is provided!")

    # sanity sort
    if config_timestamp["generation"] and \
            config_timestamp["column"]:
        time_col_name = config_timestamp["column"]
        big_raw_df = big_raw_df.sort_values(time_col_name)

    dfs = []
    if split_type == "fixed_size":
        chunk_size = math.ceil(big_raw_df.shape[0] / n_chunks)
        for chunk_id in range(n_chunks):
            df_chunk = big_raw_df.iloc[chunk_id *
                                       chunk_size:((chunk_id + 1) * chunk_size)]
            dfs.append(df_chunk)
        return dfs, chunk_size

    elif split_type == "fixed_time":
        time_evenly_spaced = np.linspace(big_raw_df[time_col_name].min(
        ), big_raw_df[time_col_name].max(), num=n_chunks + 1)
        time_evenly_spaced[-1] *= (1 + eps)

        chunk_time = (big_raw_df[time_col_name].max() -
                      big_raw_df[time_col_name].min()) / n_chunks

        for chunk_id in range(n_chunks):
            df_chunk = big_raw_df[
                (big_raw_df[time_col_name] >= time_evenly_spaced[chunk_id]) &
                (big_raw_df[time_col_name] < time_evenly_spaced[chunk_id + 1])]
            if len(df_chunk) == 0:
                print("Raw chunk_id: {}, empty df_chunk!".format(chunk_id))
                continue
            dfs.append(df_chunk)
        return dfs, chunk_time

    else:
        raise ValueError("Unknown split type")


def apply_per_field(
        df,
        config_fields,
        field_instances,
        embed_model=None):
    numpys = []
    for i, field in enumerate(config_fields):
        field_instance = field_instances[i]
        # Bit Field: (integer)
        if 'bit' in getattr(field, 'encoding', ''):
            this_numpy = df.apply(
                lambda row: field_instance.normalize(row[field.column]), axis='columns', result_type='expand').to_numpy()
        # word2vec field: (any)
        if 'word2vec' in getattr(field, 'encoding', ''):
            this_numpy = df.apply(
                lambda row: get_vector(embed_model, str(
                    row[field.column]), norm_option=True),
                axis='columns', result_type='expand').to_numpy()
        # Categorical field: (string | integer) OR Continuous Field (float)
        if 'categorical' in getattr(field, 'encoding', '') \
                or field.type == "float":
            this_numpy = field_instance.normalize(
                df[field.column].to_numpy())
        # print(field.column, this_numpy.shape)
        numpys.append(this_numpy)
    numpy_ = np.concatenate(numpys, axis=1).astype(np.float64)

    return numpys, numpy_


@ray.remote(scheduling_strategy="SPREAD", max_calls=1)
def split_per_chunk(
    config,
    metadata_fields,
    timeseries_fields,
    df_per_chunk,
    embed_model,
    global_max_flow_len,
    chunk_id,
    flowkeys_chunkidx=None,
    multi_chunk_flag=True
):
    metadata_cols = [m for m in config["metadata"]]

    # w/o DP: normalize time by per-chunk min/max
    if config["timestamp"]["generation"]:
        time_col = config["timestamp"]["column"]

        if config["timestamp"]["encoding"] == "interarrival":
            gk = df_per_chunk.groupby([m.column for m in metadata_cols])
            flow_start_list = list(gk.first()[time_col])
            assert metadata_fields[-1].name == "flow_start"
            metadata_fields[-1].min_x = float(min(flow_start_list))
            metadata_fields[-1].max_x = float(max(flow_start_list))

            interarrival_within_flow_list = list(
                gk[time_col].diff().fillna(0.0))
            assert timeseries_fields[0].name == "interarrival_within_flow"
            timeseries_fields[0].min_x = float(
                min(interarrival_within_flow_list))
            timeseries_fields[0].max_x = float(
                max(interarrival_within_flow_list))

        elif config["timestamp"]["encoding"] == "raw":
            assert timeseries_fields[0].name == time_col
            timeseries_fields[0].min_x = float(df_per_chunk[time_col].min())
            timeseries_fields[0].max_x = float(df_per_chunk[time_col].max())

    if multi_chunk_flag and flowkeys_chunkidx is None:
        raise ValueError(
            "Cross-chunk mechanism enabled, \
                cross-chunk flow stats not provided!")

    # Parse data.
    gk = df_per_chunk.groupby([m.column for m in metadata_cols])

    metadata_df = pd.DataFrame(
        list(gk.groups.keys()),
        columns=[m.column for m in metadata_cols])
    print("metadata_df:", metadata_df.shape)
    metadata_numpys, metadata_numpy = apply_per_field(
        df=metadata_df,
        config_fields=config["metadata"],
        field_instances=metadata_fields,
        embed_model=embed_model
    )
    print(f'List of metadata: '
          f'{list((k.dtype, k.shape) for k in metadata_numpys)}')
    print(f'Metadata type: {metadata_numpy.dtype}, '
          f'shape: {metadata_numpy.shape}')

    # timeseries_numpys, timeseries_numpy = apply_per_field(
    #     df=df_per_chunk,
    #     config_fields=config["timeseries"],
    #     field_instances=timeseries_fields,
    #     embed_model=embed_model
    # )
