import os
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np
import pandas as pd
from config_io import Config
from gensim.models import Word2Vec

from netshare.configs import get_config
from netshare.learn.utils.word2vec_embedding import (
    build_annoy_dictionary_word2vec,
    word2vec_train,
)
from netshare.utils.field import (
    BitField,
    ContinuousField,
    DiscreteField,
    Field,
    FieldKey,
    Normalization,
    RegexField,
    Word2VecField,
    field_config_to_key,
    key_from_field,
)
from netshare.utils.logger import logger
from netshare.utils.paths import get_preprocessed_data_folder, get_word2vec_model_path

EPS = 1e-8


class CrossChunksData(NamedTuple):
    embed_model: Optional[Word2Vec]
    flowkeys_chunkidx: Optional[dict]
    global_max_flow_len: int
    session_key_fields: Dict[FieldKey, Field]
    timeseries_fields: Dict[FieldKey, Field]


def get_flowkeys_chunkidx(df_chunks: List[pd.DataFrame]) -> Dict[str, List[int]]:
    prepost_config = get_config(
        ["pre_post_processor.config", "learn"], default_value={}
    )
    logger.info("compute flowkey-chunk list from scratch")
    session_key_config = prepost_config.get("session_key") or prepost_config["metadata"]
    flow_chunkid_keys = {}
    for chunk_id, df_chunk in enumerate(df_chunks):
        gk = df_chunk.groupby([m.column for m in session_key_config])
        flow_keys = list(gk.groups.keys())
        if len(session_key_config) == 1:
            # Solving a gotcha in pandas groupby when grouping by a single column
            flow_keys = [tuple([key]) for key in flow_keys]
        flow_keys = list(map(str, flow_keys))
        flow_chunkid_keys[chunk_id] = flow_keys

    # key: flow key
    # value: [a list of appeared chunk idx]
    flowkeys_chunkidx: Dict[str, List[int]] = {}
    for chunk_id, flowkeys in flow_chunkid_keys.items():
        logger.debug(
            "processing chunk {}/{}, # of flows: {}".format(
                chunk_id + 1, len(df_chunks), len(flowkeys)
            )
        )
        for k in flowkeys:
            if k not in flowkeys_chunkidx:
                flowkeys_chunkidx[k] = []
            flowkeys_chunkidx[k].append(chunk_id)
    return flowkeys_chunkidx


def get_global_max_flow_len(df_chunks: List[pd.DataFrame]) -> int:
    prepost_config = get_config(
        ["pre_post_processor.config", "learn"], default_value={}
    )
    session_key_config = prepost_config.get("session_key") or prepost_config["metadata"]
    if prepost_config.get("max_flow_len"):
        return int(prepost_config["max_flow_len"])

    max_flow_lens: List[int] = []
    for df_chunk in df_chunks:
        # corner case: skip for empty df_chunk
        if len(df_chunk) == 0:
            continue
        gk_chunk = df_chunk.groupby(by=[m.column for m in session_key_config])
        max_flow_lens.append(max(gk_chunk.size().values))

    return max(max_flow_lens)


def get_word2vec_model(
    df: pd.DataFrame,
) -> Tuple[Optional[Word2Vec], Optional[List[str]]]:
    word2vec_config = get_config(
        ["pre_post_processor.config", "learn.word2vec"], default_value={}
    )
    if not word2vec_config:
        logger.debug("No learn.word2vec config found, skipping the embedding model")
        return None, None
    session_key_fields = word2vec_config.get(
        "session_key", word2vec_config.get("metadata", [])
    )
    word2vec_cols = [
        m
        for m in (session_key_fields + word2vec_config.get("timeseries", []))
        if "word2vec" in m.get("encoding", "")
    ]

    if not word2vec_cols:
        logger.info("Skipping word2vec embedding: no word2vec columns")
        return None, None

    if word2vec_config["word2vec"]["pretrain_model_path"]:
        logger.info("Word2vec: Loading pretrained model")
        return (
            Word2Vec.load(word2vec_config["word2vec"]["pretrain_model_path"]),
            word2vec_cols,
        )

    logger.info("Word2vec: training model")
    os.makedirs(get_preprocessed_data_folder(), exist_ok=True)
    word2vec_train(
        df=df,
        model_name=word2vec_config["word2vec"]["model_name"],
        word2vec_cols=word2vec_cols,
        word2vec_size=word2vec_config["word2vec"]["vec_size"],
        annoy_n_trees=word2vec_config["word2vec"]["annoy_n_trees"],
    )
    word2vec_model_path = get_word2vec_model_path()

    build_annoy_dictionary_word2vec(
        df=df,
        model_path=word2vec_model_path,
        word2vec_cols=word2vec_cols,
        word2vec_size=word2vec_config["word2vec"]["vec_size"],
        n_trees=1000,
    )
    return Word2Vec.load(word2vec_model_path), word2vec_cols


def build_field_from_config(
    field: Config, df: pd.DataFrame, word2vec_cols: Optional[List[str]]
) -> Field:
    """
    This function builds the Field object from a specific field configuration.
    There are 3 types of fields:
    1. BitField: when encoding = bit. This field using the configuration n_bits.
    2. ContinuousField:
        2.1 when encoding = word2vec, we use the word2vec model to encode the field.
        2.2 when type = float, we take the min and max value of the column in the dataframe.
    3. DiscreteField: when encoding = categorical or type = string, we use the unique values of the column in the dataframe.
    """
    prepost_config = get_config("pre_post_processor.config", default_value={})

    if not isinstance(field.get("column"), str) and not isinstance(
        field.get("columns"), list
    ):
        raise ValueError(
            f'Malformed field configuration: "column" should be a string or "columns" a list of strings ({field})'
        )
    if "type" not in field:
        raise ValueError(f'Malformed field configuration: Missing "type" in: {field}')

    field_name: Union[str, List[str]] = (
        field.get("name", "") or field.get("column", "") or field["columns"]
    )
    if field.get("column", ""):
        this_df = df[field["column"]]
    else:
        this_df = df[field["columns"]].stack()

    # Bit Field: (integer)
    field_instance: Field
    if "bit" in field.get("encoding", ""):
        if field["type"] != "integer":
            raise ValueError('"encoding=bit" can be only used for "type=integer"')
        if "n_bits" not in field:
            raise ValueError("`n_bits` needs to be specified for bit fields")
        return BitField(name=field_name, num_bits=field["n_bits"])

    # word2vec field: (any)
    elif "word2vec" in field.get("encoding", ""):
        return Word2VecField(
            name=field_name,
            word2vec_size=prepost_config["word2vec"]["vec_size"],
            word2vec_cols=word2vec_cols,
        )

    # Categorical field: (string | integer)
    elif "categorical" in field.get("encoding", "") or field["type"] == "string":
        if field["type"] not in ["string", "integer"]:
            raise ValueError(
                '"encoding=cateogrical" can be only used for "type=(string | integer)"'
            )
        if "regex" in field:
            return RegexField(
                regex=field["regex"],
                choices=[],  # Note: it will be filled *per chunk* in the per-chunk preprocessing
                name=field_name,
            )
        return DiscreteField(
            choices=list(pd.unique(this_df)),
            name=field_name,
        )

    # Continuous Field: (float)
    elif field["type"] == "float":
        return ContinuousField(
            name=field_name,
            log1p_norm=field.get("log1p_norm", False),
            norm_option=Normalization.from_config(field["normalization"]),
            min_x=min(this_df) - EPS,
            max_x=max(this_df) + EPS,
            dim_x=1,
        )

    raise ValueError(
        "Unable to build field from config (known field type / encoding): {}".format(
            field
        )
    )


def build_fields(
    df: pd.DataFrame, word2vec_cols: Optional[List[str]]
) -> Tuple[Dict[FieldKey, Field], Dict[FieldKey, Field]]:

    session_key_fields: Dict[FieldKey, Field] = {
        field_config_to_key(field): build_field_from_config(field, df, word2vec_cols)
        for field in get_config(
            ["pre_post_processor.config.metadata", "learn.session_key"]
        )
    }
    timeseries_fields: Dict[FieldKey, Field] = {
        field_config_to_key(field): build_field_from_config(field, df, word2vec_cols)
        for field in get_config(
            ["pre_post_processor.config.timeseries", "learn.timeseries"]
        )
    }

    if get_config("global_config.n_chunks", default_value=1) > 1:
        new_field = DiscreteField(name="startFromThisChunk", choices=[0.0, 1.0])
        session_key_fields[key_from_field(new_field)] = new_field

        for chunk_id in range(get_config("global_config.n_chunks")):
            new_field = DiscreteField(
                name="chunk_{}".format(chunk_id), choices=[0.0, 1.0]
            )
            session_key_fields[key_from_field(new_field)] = new_field

    return session_key_fields, timeseries_fields


def setup_cross_chunks_data(
    big_df: pd.DataFrame, df_chunks: List[pd.DataFrame]
) -> CrossChunksData:
    """
    This function splits the input data into chunks, and compute the flow tags for each chunk.
    """
    embed_model, word2vec_cols = get_word2vec_model(big_df)
    session_key_fields, timeseries_fields = build_fields(big_df, word2vec_cols)
    flowkeys_chunkidx = get_flowkeys_chunkidx(df_chunks)
    global_max_flow_len = get_global_max_flow_len(df_chunks)

    return CrossChunksData(
        embed_model=embed_model,
        flowkeys_chunkidx=flowkeys_chunkidx,
        global_max_flow_len=global_max_flow_len,
        session_key_fields=session_key_fields,
        timeseries_fields=timeseries_fields,
    )
