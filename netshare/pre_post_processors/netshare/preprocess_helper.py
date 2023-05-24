import os
import math
import copy
import pickle
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

    if n_chunks > 1 and (
        (not config_timestamp["column"]) or (
            not config_timestamp["generation"])):
        raise ValueError(
            "Trying to split into multiple chunks by timestamp but no timestamp is provided!")

    # sanity sort
    if config_timestamp["generation"] and \
            config_timestamp["column"]:
        time_col_name = config_timestamp["column"]
        big_raw_df = big_raw_df.sort_values(time_col_name)

    if n_chunks == 1:

        chunk_time = (big_raw_df[time_col_name].max() -
                      big_raw_df[time_col_name].min()) / n_chunks
        return [big_raw_df], chunk_time

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
        original_df,
        config_fields,
        field_instances,
        embed_model=None):
    original_df.reset_index(drop=True, inplace=True)
    new_df = copy.deepcopy(original_df)
    new_field_list = []
    for i, field in enumerate(config_fields):
        field_instance = field_instances[i]
        # Bit Field: (integer)
        if 'bit' in getattr(field, 'encoding', ''):
            this_df = original_df.apply(lambda row: field_instance.normalize(
                row[field.column]), axis='columns', result_type='expand')
            this_df.columns = [
                f'{field.column}_{i}' for i in range(this_df.shape[1])]
            new_field_list += list(this_df.columns)
            new_df = pd.concat([new_df, this_df], axis=1)

        # word2vec field: (any)
        if 'word2vec' in getattr(field, 'encoding', ''):
            this_df = original_df.apply(
                lambda row: get_vector(embed_model, str(
                    row[field.column]), norm_option=True),
                axis='columns', result_type='expand')
            this_df.columns = [
                f'{field.column}_{i}' for i in range(this_df.shape[1])]
            new_field_list += list(this_df.columns)
            new_df = pd.concat([new_df, this_df], axis=1)

        # Categorical field: (string | integer)
        if 'categorical' in getattr(field, 'encoding', ''):
            this_df = pd.DataFrame(field_instance.normalize(
                original_df[field.column].to_numpy()))
            this_df.columns = [
                f'{field.column}_{i}' for i in range(this_df.shape[1])]
            new_field_list += list(this_df.columns)
            new_df = pd.concat([new_df, this_df], axis=1)

        # Continuous Field: (float)
        if field.type == "float":
            new_df[field.column] = field_instance.normalize(
                original_df[field.column].to_numpy().reshape(-1, 1))
            new_field_list.append(field.column)

    return new_df, new_field_list


@ray.remote(scheduling_strategy="SPREAD", max_calls=1)
def split_per_chunk(
    config,
    metadata_fields,
    timeseries_fields,
    df_per_chunk,
    embed_model,
    global_max_flow_len,
    chunk_id,
    data_out_dir,
    flowkeys_chunkidx=None,
):
    split_name = config["split_name"]
    metadata_cols = [m for m in config["metadata"]]

    # Truncate groups with length greater than global_max_flow_len
    def process_group(group):
        if len(group) > global_max_flow_len:
            processed_group = group.head(global_max_flow_len)
        else:
            processed_group = group
        return processed_group
    
    def truncate_group(raw_df, metadata_cols):
        grouped = raw_df.groupby([m.column for m in metadata_cols])
        processed = grouped.apply(process_group)

        # reset the index of the resulting DataFrame
        processed = processed.reset_index(drop=True)

        return processed
    
    print("Before truncation, df_per_chunk:", df_per_chunk.shape)
    df_per_chunk = truncate_group(df_per_chunk, metadata_cols)
    print("After truncation, df_per_chunk:", df_per_chunk.shape)

    df_per_chunk, new_metadata_list = apply_per_field(
        original_df=df_per_chunk,
        config_fields=config["metadata"],
        field_instances=metadata_fields,
        embed_model=embed_model
    )

    df_per_chunk, new_timeseries_list = apply_per_field(
        original_df=df_per_chunk,
        config_fields=config["timeseries"],
        field_instances=timeseries_fields,
        embed_model=embed_model
    )
    print("df_per_chunk:", df_per_chunk.shape)

    # Multi-chunk related field instances
    # n_chunk=1 reduces to plain DoppelGANger
    if config["n_chunks"] > 1:
        metadata_fields.append(DiscreteField(
            name="startFromThisChunk",
            choices=[0.0, 1.0]
        ))

        for i in range(config["n_chunks"]):
            metadata_fields.append(DiscreteField(
                name="chunk_{}".format(i),
                choices=[0.0, 1.0]
            ))

    # w/o DP: normalize time by per-chunk min/max
    if config["timestamp"]["generation"]:
        if "column" not in config["timestamp"]:
            raise ValueError(
                'Timestamp generation is enabled! "column" must be set...')
        time_col = config["timestamp"]["column"]

        if config["timestamp"]["encoding"] == "interarrival":
            gk = df_per_chunk.groupby(new_metadata_list)
            flow_start_list = list(gk.first()[time_col])
            metadata_fields.append(ContinuousField(
                name="flow_start",
                norm_option=getattr(
                    Normalization, config["timestamp"].normalization),
                min_x=min(flow_start_list),
                max_x=max(flow_start_list)
            ))
            flow_start_list = metadata_fields[-1].normalize(
                np.array(flow_start_list).reshape(-1, 1))

            interarrival_within_flow_list = list(
                gk[time_col].diff().fillna(0.0))
            df_per_chunk["interarrival_within_flow"] = interarrival_within_flow_list
            timeseries_fields.insert(0, ContinuousField(
                name="interarrival_within_flow",
                norm_option=getattr(
                    Normalization, config["timestamp"].normalization),
                min_x=min(interarrival_within_flow_list),
                max_x=max(interarrival_within_flow_list)
            ))
            df_per_chunk["interarrival_within_flow"] = timeseries_fields[0].normalize(
                df_per_chunk["interarrival_within_flow"].to_numpy().reshape(-1, 1))
            new_timeseries_list.insert(0, "interarrival_within_flow")

        elif config["timestamp"]["encoding"] == "raw":
            timeseries_fields.insert(0, ContinuousField(
                name=getattr(
                    config["timestamp"], "name", config["timestamp"]["column"]),
                norm_option=getattr(
                    Normalization, config["timestamp"].normalization),
                min_x=min(df_per_chunk[time_col]),
                max_x=max(df_per_chunk[time_col])
            ))
            df_per_chunk[time_col] = timeseries_fields[0].normalize(
                df_per_chunk[time_col].to_numpy().reshape(-1, 1)
            )
            new_timeseries_list.insert(0, time_col)
        else:
            raise ValueError("Timestamp encoding can be only \
            `interarrival` or 'raw")

    # print("new_metadata_list:", new_metadata_list)
    # print("new_timeseries_list:", new_timeseries_list)

    gk = df_per_chunk.groupby(new_metadata_list)
    data_attribute = np.array(list(gk.groups.keys()))
    data_feature = []
    data_gen_flag = []
    flow_tags = []
    for group_name, df_group in tqdm(gk):
        # RESET INDEX TO MAKE IT START FROM ZERO
        df_group = df_group.reset_index(drop=True)
        data_feature.append(df_group[new_timeseries_list].to_numpy())
        data_gen_flag.append(np.ones((len(df_group),), dtype=float) * 1.0)

        attr_per_row = []
        if config["n_chunks"] > 1:
            if flowkeys_chunkidx is None:
                raise ValueError(
                    "Cross-chunk mechanism enabled, \
                    cross-chunk flow stats not provided!")
            ori_group_name = tuple(
                df_group.iloc[0][[m.column for m in config["metadata"]]])

            # MULTI-CHUNK TAGS: TO BE OPTIMIZED FOR PERFORMANCE
            if str(ori_group_name) in flowkeys_chunkidx:  # sanity check
                # flow starts from this chunk
                if flowkeys_chunkidx[str(ori_group_name)][0] == chunk_id:
                    # attr_per_row += list(
                    # fields_dict["startFromThisChunk"].normalize(1.0))
                    attr_per_row += [0.0, 1.0]
                    # num_flows_startFromThisChunk += 1

                    for i in range(config["n_chunks"]):
                        if i in flowkeys_chunkidx[str(ori_group_name)]:
                            # attr_per_row += list(fields_dict["chunk_{}".format(
                            #     i)].normalize(1.0))
                            attr_per_row += [0.0, 1.0]
                        else:
                            # attr_per_row += list(fields_dict["chunk_{}".format(
                            #     i)].normalize(0.0))
                            attr_per_row += [1.0, 0.0]

                # flow does not start from this chunk
                else:
                    # attr_per_row += list(
                    #     fields_dict["startFromThisChunk"].normalize(0.0))
                    attr_per_row += [1.0, 0.0]
                    if split_name == "multichunk_dep_v1":
                        for i in range(config["n_chunks"]):
                            # attr_per_row += list(fields_dict["chunk_{}".format(
                            #     i)].normalize(0.0))
                            attr_per_row += [1.0, 0.0]

                    elif split_name == "multichunk_dep_v2":
                        for i in range(config["n_chunks"]):
                            if i in flowkeys_chunkidx[str(ori_group_name)]:
                                # attr_per_row += list(
                                #     fields_dict["chunk_{}".format(i)].normalize(1.0))
                                attr_per_row += [0.0, 1.0]
                            else:
                                # attr_per_row += list(
                                #     fields_dict["chunk_{}".format(i)].normalize(0.0))
                                attr_per_row += [1.0, 0.0]
                flow_tags.append(attr_per_row)
            else:
                raise ValueError(
                    f"{ori_group_name} not found in the raw file!")
    if config["n_chunks"] > 1:
        data_attribute = np.concatenate(
            (data_attribute, np.array(flow_tags)), axis=1)
    if config["timestamp"]["generation"] and \
            config["timestamp"]["encoding"] == "interarrival":
        data_attribute = np.concatenate(
            (data_attribute, np.array(flow_start_list).reshape(-1, 1)), axis=1)

    data_attribute = np.asarray(data_attribute)
    data_feature = np.stack(
        [np.pad(
            arr,
            ((0, global_max_flow_len - arr.shape[0]), (0, 0)),
            mode='constant',
            constant_values=0) for arr in data_feature])
    data_gen_flag = np.stack([np.pad(
        arr,
        ((0, global_max_flow_len - arr.shape[0])),
        mode='constant',
        constant_values=0) for arr in data_gen_flag])
    print("data_attribute: {}, {}GB in memory".format(
        np.shape(data_attribute),
        data_attribute.size * data_attribute.itemsize / (10**9)))
    print("data_feature: {}, {}GB in memory".format(
        np.shape(data_feature),
        data_feature.size * data_feature.itemsize / (10**9)))
    print("data_gen_flag: {}, {}GB in memory".format(
        np.shape(data_gen_flag),
        data_gen_flag.size * data_gen_flag.itemsize / (10**9)))

    # Write files
    os.makedirs(data_out_dir, exist_ok=True)
    df_per_chunk.to_csv(os.path.join(data_out_dir, "raw.csv"), index=False)
    np.savez(
        os.path.join(data_out_dir, "data_train.npz"),
        data_attribute=data_attribute,
        data_feature=data_feature,
        data_gen_flag=data_gen_flag
    )

    with open(os.path.join(
            data_out_dir, 'data_attribute_output.pkl'), 'wb') as f:
        data_attribute_output = []
        for v in metadata_fields:
            if isinstance(v, BitField):
                data_attribute_output += v.getOutputType()
            else:
                data_attribute_output.append(v.getOutputType())
        pickle.dump(data_attribute_output, f)
    with open(os.path.join(
            data_out_dir, 'data_feature_output.pkl'), 'wb') as f:
        data_feature_output = []
        for v in timeseries_fields:
            if isinstance(v, BitField):
                data_feature_output += v.getOutputType()
            else:
                data_feature_output.append(v.getOutputType())
        pickle.dump(data_feature_output, f)
    with open(os.path.join(
            data_out_dir, 'data_attribute_fields.pkl'), 'wb') as f:
        pickle.dump(metadata_fields, f)
    with open(os.path.join(
            data_out_dir, 'data_feature_fields.pkl'), 'wb') as f:
        pickle.dump(timeseries_fields, f)
