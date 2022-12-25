import os
import copy
import pickle
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm
from gensim.models import Word2Vec
from config_io import Config

import netshare.ray as ray
from netshare.configs import set_config, get_config
from netshare.pre_process.field import FieldKey, field_config_to_key, key_from_field
from netshare.pre_process.prepare_cross_chunks_data import CrossChunksData
from netshare.utils import (
    Field,
    BitField,
    Normalization,
    ContinuousField,
)
from netshare.pre_post_processors.netshare.embedding_helper import get_vector


def get_chunk_dir(target_dir: str, chunk_id: int) -> str:
    return os.path.join(target_dir, f"chunkid-{chunk_id}")


def apply_per_field(
    original_df: pd.DataFrame,
    config_fields: Config,
    field_instances: Dict[FieldKey, Field],
    embed_model: Optional[Word2Vec] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    original_df.reset_index(drop=True, inplace=True)
    new_df = copy.deepcopy(original_df)
    new_field_list: List[str] = []
    for i, field in enumerate(config_fields):
        field_instance = field_instances[field_config_to_key(field)]
        # Bit Field: (integer)
        if "bit" in getattr(field, "encoding", ""):
            this_df = original_df.apply(
                lambda row: field_instance.normalize(row[field.column]),
                axis="columns",
                result_type="expand",
            )
            this_df.columns = [f"{field.column}_{i}" for i in range(this_df.shape[1])]
            new_field_list += list(this_df.columns)
            new_df = pd.concat([new_df, this_df], axis=1)

        # word2vec field: (any)
        if "word2vec" in getattr(field, "encoding", ""):
            this_df = original_df.apply(
                lambda row: get_vector(
                    embed_model, str(row[field.column]), norm_option=True
                ),
                axis="columns",
                result_type="expand",
            )
            this_df.columns = [f"{field.column}_{i}" for i in range(this_df.shape[1])]
            new_field_list += list(this_df.columns)
            new_df = pd.concat([new_df, this_df], axis=1)

        # Categorical field: (string | integer)
        if "categorical" in getattr(field, "encoding", ""):
            this_df = pd.DataFrame(
                field_instance.normalize(original_df[field.column].to_numpy())
            )
            this_df.columns = [f"{field.column}_{i}" for i in range(this_df.shape[1])]
            new_field_list += list(this_df.columns)
            new_df = pd.concat([new_df, this_df], axis=1)

        # Continuous Field: (float)
        if field.type == "float":
            new_df[field.column] = field_instance.normalize(
                original_df[field.column].to_numpy().reshape(-1, 1)
            )
            new_field_list.append(field.column)

    return new_df, new_field_list


def write_chunk_data(
    df_per_chunk: pd.DataFrame,
    cross_chunks_data: CrossChunksData,
    target_dir: str,
    data_attribute: np.array,
    data_gen_flag: np.array,
    data_feature: np.array,
    chunk_id: int,
) -> None:
    data_out_dir = get_chunk_dir(target_dir, chunk_id=chunk_id)
    os.makedirs(data_out_dir, exist_ok=True)
    df_per_chunk.to_csv(os.path.join(data_out_dir, "raw.csv"), index=False)

    num_rows = data_attribute.shape[0]
    os.makedirs(os.path.join(data_out_dir, "data_train_npz"), exist_ok=True)
    gt_lengths = []
    for row_id in range(num_rows):
        gt_lengths.append(sum(data_gen_flag[row_id]))
        np.savez(
            os.path.join(data_out_dir, "data_train_npz", f"data_train_{row_id}.npz"),
            data_feature=data_feature[row_id],
            data_attribute=data_attribute[row_id],
            data_gen_flag=data_gen_flag[row_id],
            global_max_flow_len=[cross_chunks_data.global_max_flow_len],
        )
    np.save(os.path.join(data_out_dir, "gt_lengths"), gt_lengths)

    with open(os.path.join(data_out_dir, "data_attribute_output.pkl"), "wb") as f:
        data_attribute_output = []
        for v in cross_chunks_data.metadata_fields.values():
            if isinstance(v, BitField):
                data_attribute_output += v.getOutputType()
            else:
                data_attribute_output.append(v.getOutputType())
        pickle.dump(data_attribute_output, f)
    with open(os.path.join(data_out_dir, "data_feature_output.pkl"), "wb") as f:
        data_feature_output = []
        for v in cross_chunks_data.timeseries_fields.values():
            if isinstance(v, BitField):
                data_feature_output += v.getOutputType()
            else:
                data_feature_output.append(v.getOutputType())
        pickle.dump(data_feature_output, f)
    with open(os.path.join(data_out_dir, "data_attribute_fields.pkl"), "wb") as f:
        pickle.dump(cross_chunks_data.metadata_fields, f)
    with open(os.path.join(data_out_dir, "data_feature_fields.pkl"), "wb") as f:
        pickle.dump(cross_chunks_data.timeseries_fields, f)


@ray.remote(scheduling_strategy="SPREAD", max_calls=1)
def preprocess_per_chunk(
    config: Config,
    cross_chunks_data: CrossChunksData,
    df_per_chunk: pd.DataFrame,
    chunk_id: int,
    target_dir: str,
) -> None:
    set_config(config)
    metadata_config = get_config("pre_post_processor.config.metadata")
    timeseries_config = get_config("pre_post_processor.config.timeseries")
    split_name = get_config("pre_post_processor.config.split_name")
    timestamp_config = get_config("pre_post_processor.config.timestamp")

    metadata_cols = [m for m in metadata_config]

    df_per_chunk, new_metadata_list = apply_per_field(
        original_df=df_per_chunk,
        config_fields=metadata_config,
        field_instances=cross_chunks_data.metadata_fields,
        embed_model=cross_chunks_data.embed_model,
    )

    df_per_chunk, new_timeseries_list = apply_per_field(
        original_df=df_per_chunk,
        config_fields=timeseries_config,
        field_instances=cross_chunks_data.timeseries_fields,
        embed_model=cross_chunks_data.embed_model,
    )

    # w/o DP: normalize time by per-chunk min/max
    if timestamp_config["generation"]:
        if "column" not in timestamp_config:
            raise ValueError('Timestamp generation is enabled! "column" must be set...')
        time_col = timestamp_config["column"]

        if timestamp_config["encoding"] == "interarrival":
            gk = df_per_chunk.groupby([m.column for m in metadata_cols])
            flow_start_list = list(gk.first()[time_col])
            flow_start_metadata_field = ContinuousField(
                name="flow_start",
                norm_option=getattr(Normalization, timestamp_config.normalization),
                min_x=min(flow_start_list),
                max_x=max(flow_start_list),
            )

            cross_chunks_data.metadata_fields[
                key_from_field(flow_start_metadata_field)
            ] = flow_start_metadata_field
            flow_start_list = flow_start_metadata_field.normalize(
                np.array(flow_start_list).reshape(-1, 1)
            )

            interarrival_within_flow_list = list(gk[time_col].diff().fillna(0.0))
            df_per_chunk["interarrival_within_flow"] = interarrival_within_flow_list
            interarrival_within_flow_timeseries_field = ContinuousField(
                name="interarrival_within_flow",
                norm_option=getattr(Normalization, timestamp_config.normalization),
                min_x=min(interarrival_within_flow_list),
                max_x=max(interarrival_within_flow_list),
            )
            cross_chunks_data.timeseries_fields[
                key_from_field(interarrival_within_flow_timeseries_field)
            ] = interarrival_within_flow_timeseries_field
            df_per_chunk[
                "interarrival_within_flow"
            ] = interarrival_within_flow_timeseries_field.normalize(
                df_per_chunk["interarrival_within_flow"].to_numpy().reshape(-1, 1)
            )
            new_timeseries_list.insert(0, "interarrival_within_flow")

        elif timestamp_config["encoding"] == "raw":
            timestamp_field = ContinuousField(
                name=getattr(timestamp_config, "name", timestamp_config["column"]),
                norm_option=getattr(Normalization, timestamp_config.normalization),
                min_x=min(df_per_chunk[time_col]),
                max_x=max(df_per_chunk[time_col]),
            )
            cross_chunks_data.timeseries_fields[
                key_from_field(timestamp_field)
            ] = timestamp_field
            df_per_chunk[time_col] = timestamp_field.normalize(
                df_per_chunk[time_col].to_numpy().reshape(-1, 1)
            )
            new_timeseries_list.insert(0, time_col)
        else:
            raise ValueError("Timestamp encoding can be only 'interarrival' or 'raw'")

    gk = df_per_chunk.groupby(new_metadata_list)
    data_attribute: np.array = np.array(list(gk.groups.keys()))
    data_feature = []
    data_gen_flag: List[List[float]] = []
    flow_tags: List[List[float]] = []
    for group_name, df_group in tqdm(gk):
        # RESET INDEX TO MAKE IT START FROM ZERO
        df_group = df_group.reset_index(drop=True)
        data_feature.append(df_group[new_timeseries_list].to_numpy())
        data_gen_flag.append([1.0] * len(df_group))

        attr_per_row: List[float] = []
        if get_config("global_config.n_chunks") > 1:
            if cross_chunks_data.flowkeys_chunkidx is None:
                raise ValueError(
                    "Cross-chunk mechanism enabled, \
                    cross-chunk flow stats not provided!"
                )
            ori_group_name = tuple(
                df_group.iloc[0][[m.column for m in metadata_config]]
            )

            # MULTI-CHUNK TAGS: TO BE OPTIMIZED FOR PERFORMANCE
            if (
                str(ori_group_name) in cross_chunks_data.flowkeys_chunkidx
            ):  # sanity check
                # flow starts from this chunk
                if (
                    cross_chunks_data.flowkeys_chunkidx[str(ori_group_name)][0]
                    == chunk_id
                ):
                    attr_per_row += [0.0, 1.0]

                    for i in range(get_config("global_config.n_chunks")):
                        if (
                            i
                            in cross_chunks_data.flowkeys_chunkidx[str(ori_group_name)]
                        ):
                            attr_per_row += [0.0, 1.0]
                        else:
                            attr_per_row += [1.0, 0.0]

                # flow does not start from this chunk
                else:
                    attr_per_row += [1.0, 0.0]
                    if split_name == "multichunk_dep_v1":
                        for i in range(get_config("global_config.n_chunks")):
                            attr_per_row += [1.0, 0.0]

                    elif split_name == "multichunk_dep_v2":
                        for i in range(get_config("global_config.n_chunks")):
                            if (
                                i
                                in cross_chunks_data.flowkeys_chunkidx[
                                    str(ori_group_name)
                                ]
                            ):
                                attr_per_row += [0.0, 1.0]
                            else:
                                attr_per_row += [1.0, 0.0]
                flow_tags.append(attr_per_row)
            else:
                raise ValueError(f"{ori_group_name} not found in the raw file!")
    if get_config("global_config.n_chunks") > 1:
        data_attribute = np.concatenate((data_attribute, np.array(flow_tags)), axis=1)
    if (
        timestamp_config["generation"]
        and timestamp_config["encoding"] == "interarrival"
    ):
        data_attribute = np.concatenate(
            (data_attribute, np.array(flow_start_list).reshape(-1, 1)), axis=1
        )

    data_attribute = np.asarray(data_attribute)
    data_feature = np.asarray(data_feature)
    data_gen_flag = np.asarray(data_gen_flag)

    write_chunk_data(
        df_per_chunk=df_per_chunk,
        cross_chunks_data=cross_chunks_data,
        target_dir=target_dir,
        data_attribute=data_attribute,
        data_gen_flag=data_gen_flag,
        data_feature=data_feature,
        chunk_id=chunk_id,
    )
