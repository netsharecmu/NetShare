import copy
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from config_io import Config
from gensim.models import Word2Vec
from pandas.core.groupby import DataFrameGroupBy

import netshare.utils.ray as ray
from netshare.configs import get_config
from netshare.preprocess import preprocess_api
from netshare.preprocess.prepare_cross_chunks_data import CrossChunksData
from netshare.preprocess.utils.embedding_helper import get_vector
from netshare.utils.field import (
    ContinuousField,
    Field,
    FieldKey,
    Normalization,
    field_config_to_key,
    key_from_field,
)


def apply_configuration_fields(
    original_df: pd.DataFrame,
    config_fields: Config,
    field_instances: Dict[FieldKey, Field],
    embed_model: Optional[Word2Vec] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    This function applies the fields configuration to the original dataframe.
    It executes Field.normalize on the relevant columns and tracks the new columns that were added to the dataframe.

    :return: the new dataframe and the list of the new columns.
    """
    original_df.reset_index(drop=True, inplace=True)
    new_df = copy.deepcopy(original_df)
    new_field_list: List[str] = []
    for field in config_fields:
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
            for column in field.get("columns") or [field.column]:
                new_df[column] = field_instance.normalize(
                    original_df[column].to_numpy().reshape(-1, 1)
                )
                new_field_list.append(column)

        if "regex" in field:
            for column in field.get("columns") or [field.column]:
                normalized_df = pd.DataFrame(
                    field_instance.normalize(
                        original_df[column].to_numpy().reshape(-1, 1)
                    )
                )
                normalized_df.columns = [
                    f"{field_instance.name}_{i}" for i in range(normalized_df.shape[1])
                ]
                new_field_list += list(normalized_df.columns)
                new_df = pd.concat([new_df, normalized_df], axis=1)

    return new_df, new_field_list


def write_chunk_data(
    df_per_chunk: pd.DataFrame,
    cross_chunks_data: CrossChunksData,
    data_attribute: np.array,
    data_feature: np.array,
    chunk_id: int,
) -> None:
    """
    This function writes the data of a single chunk using the preprocess API.

    TODO: Why do we store the cross_chunks_data for every chunk?
        It creates trouble later when we try to load it in the postprocess phase.
    """
    preprocess_api.create_dirs(chunk_id)
    preprocess_api.write_raw_chunk(df_per_chunk, chunk_id)
    preprocess_api.write_data_train_npz(
        data_attribute,
        data_feature,
        cross_chunks_data.global_max_flow_len,
        chunk_id,
    )
    preprocess_api.write_attributes(cross_chunks_data.metadata_fields, chunk_id)
    preprocess_api.write_features(cross_chunks_data.timeseries_fields, chunk_id)


def apply_timestamp_generation(
    df_per_chunk: pd.DataFrame,
    cross_chunks_data: CrossChunksData,
    new_timeseries_list: List[str],
) -> np.array:
    """
    This function reads the configuration from pre_post_processor.config.timestamp,
        and generate the normalized timestamp column if needed.

    There are two types of timestamp generation:
    1. encoding = raw: Taking a specific column as another timestamp column. TODO: I don't understand why this is needed.
    2. encoding = interarrival: For each tuple of metadata, we take the start time of the flow a metadata column,
        and also take the diff between every packet to the previous one as a feature column.
    """
    metadata_config = get_config(
        "pre_post_processor.config.metadata", path2="preprocess.metadata"
    )
    timestamp_config = get_config(
        "pre_post_processor.config.timestamp",
        path2="preprocess.timestamp",
        default_value={},
    )

    metadata_cols = [m for m in metadata_config]

    additional_data_attributes = np.array([])

    # w/o DP: normalize time by per-chunk min/max
    if timestamp_config.get("generation"):
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
            additional_data_attributes = np.array(flow_start_list).reshape(-1, 1)

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

    return additional_data_attributes


def apply_cross_chunk_mechanism(
    df_group: DataFrameGroupBy,
    cross_chunks_data: CrossChunksData,
    chunk_id: int,
) -> List[float]:
    """
    This function executes the cross-chunk mechanism (if global_config.n_chunks > 1) for
        the given group of metadata attributes.
    In this mechanism, we use the data in cross_chunks_data.flowkeys_chunkidx to compute the
        flow-tags attributes for the given group, and return it.

    TODO - why is it happen in every chunk? Probably the groups are distributed evenly across every chunk.
        Maybe we can spare performance here?
        Note the signature of this function - it doesn't get the data of the current chunk...
    """
    metadata_config = get_config(
        "pre_post_processor.config.metadata", path2="preprocess.metadata"
    )

    attr_per_row: List[float] = []
    if get_config("global_config.n_chunks", default_value=1) > 1:
        split_name = get_config(
            "pre_post_processor.config.split_name", path2="preprocess.split_name"
        )
        if cross_chunks_data.flowkeys_chunkidx is None:
            raise ValueError(
                "Internal error: cross-chunk data is missing is the per-chunk processing!"
            )
        ori_group_name = tuple(df_group.iloc[0][[m.column for m in metadata_config]])
        chunk_indexes = cross_chunks_data.flowkeys_chunkidx.get(str(ori_group_name))

        # MULTI-CHUNK TAGS: TO BE OPTIMIZED FOR PERFORMANCE
        if chunk_indexes:  # sanity check
            # flow starts from this chunk
            if chunk_indexes[0] == chunk_id:
                attr_per_row += [0.0, 1.0]

                for i in range(get_config("global_config.n_chunks")):
                    if i in chunk_indexes:
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
                        if i in chunk_indexes:
                            attr_per_row += [0.0, 1.0]
                        else:
                            attr_per_row += [1.0, 0.0]
        else:
            raise ValueError(f"{ori_group_name} not found in the raw file!")
    return attr_per_row


def reduce_samples(
    data_attribute: np.ndarray, data_feature: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    This function reduces the number of samples in the given dataframe, according to the
        global_config.sample_ratio.
    """
    samples = get_config(
        "pre_post_processor.config.num_train_samples", default_value=False
    )
    # TODO: I use the should_sample because the old code expecting different data
    #   formats between the different PrePostProcessor
    should_sample = (
        get_config("pre_post_processor.class", default_value="")
        == "DGRowPerSamplePrePostProcessor"
    )
    if samples and should_sample:
        np.random.seed(get_config("global_config.seed", default_value=0))
        ids = np.random.permutation(data_attribute.shape[0])
        data_attribute = data_attribute[ids[:samples]]
        data_feature = np.concatenate(data_feature)[ids[:samples]]
        data_feature = data_feature.reshape(
            (data_feature.shape[0], data_feature.shape[1], 1)
        )
    return data_attribute, data_feature


@ray.remote(scheduling_strategy="SPREAD", max_calls=1)
def preprocess_per_chunk(
    cross_chunks_data: CrossChunksData,
    df_per_chunk: pd.DataFrame,
    chunk_id: int,
) -> None:
    metadata_config = get_config(
        "pre_post_processor.config.metadata", path2="preprocess.metadata"
    )
    timeseries_config = get_config(
        "pre_post_processor.config.timeseries", path2="preprocess.timeseries"
    )

    df_per_chunk, new_metadata_list = apply_configuration_fields(
        original_df=df_per_chunk,
        config_fields=metadata_config,
        field_instances=cross_chunks_data.metadata_fields,
        embed_model=cross_chunks_data.embed_model,
    )

    df_per_chunk, new_timeseries_list = apply_configuration_fields(
        original_df=df_per_chunk,
        config_fields=timeseries_config,
        field_instances=cross_chunks_data.timeseries_fields,
        embed_model=cross_chunks_data.embed_model,
    )

    timestamp_generation_attributes = apply_timestamp_generation(
        df_per_chunk, cross_chunks_data, new_timeseries_list
    )

    gk = df_per_chunk.groupby(new_metadata_list)
    data_feature_list: List[np.ndarray] = []
    flow_tags: List[List[float]] = []
    for group_name, df_group in gk:
        df_group = df_group.reset_index(
            drop=True
        )  # reset index to make it start from zero
        data_feature_list.append(df_group[new_timeseries_list].to_numpy())
        attr_per_row = apply_cross_chunk_mechanism(
            df_group=df_group, cross_chunks_data=cross_chunks_data, chunk_id=chunk_id
        )
        if attr_per_row:
            flow_tags.append(attr_per_row)

    data_attribute: np.array
    if (
        get_config("preprocess.attributes_from_data", default_value=False)
        or get_config("pre_post_processor.class", default_value="")
        == "DGRowPerSamplePrePostProcessor"
    ):
        data_attribute = df_per_chunk[new_metadata_list].to_numpy()
    else:
        data_attribute = np.array(list(gk.groups.keys()))
        if flow_tags:
            data_attribute = np.concatenate(
                (data_attribute, np.array(flow_tags)), axis=1
            )
        if len(timestamp_generation_attributes) > 0:
            data_attribute = np.concatenate(
                (data_attribute, timestamp_generation_attributes), axis=1
            )

    data_attribute = np.asarray(data_attribute)
    data_feature = np.asarray(data_feature_list)

    data_attribute, data_feature = reduce_samples(data_attribute, data_feature)

    write_chunk_data(
        df_per_chunk=df_per_chunk,
        cross_chunks_data=cross_chunks_data,
        data_attribute=data_attribute,
        data_feature=data_feature,
        chunk_id=chunk_id,
    )
