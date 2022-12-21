from typing import List, Dict, NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd
from gensim.models import Word2Vec

from netshare.logger import logger
from netshare.pre_post_processors.netshare.word2vec_embedding import word2vec_train
from netshare.utils import (
    BitField,
    ContinuousField,
    Normalization,
    DiscreteField,
    Field,
)


EPS = 1e-8


class CrossChunkData(NamedTuple):
    embed_model: Optional[Word2Vec]
    flowkeys_chunkidx: dict
    global_max_flow_len: int
    metadata_fields: List[Field]
    timeseries_fields: List[Field]


def get_flowkeys_chunkidx(
    df_chunks: List[pd.DataFrame], config: dict
) -> Dict[str, List[int]]:
    prepost_config = config.get("pre_post_processor", {}).get("config", {})
    print("compute flowkey-chunk list from scratch...")
    flow_chunkid_keys = {}
    for chunk_id, df_chunk in enumerate(df_chunks):
        gk = df_chunk.groupby([m.column for m in prepost_config["metadata"]])
        flow_keys = list(gk.groups.keys())
        flow_keys = list(map(str, flow_keys))
        flow_chunkid_keys[chunk_id] = flow_keys

    # key: flow key
    # value: [a list of appeared chunk idx]
    flowkeys_chunkidx: Dict[str, List[int]] = {}
    for chunk_id, flowkeys in flow_chunkid_keys.items():
        print(
            "processing chunk {}/{}, # of flows: {}".format(
                chunk_id + 1, len(df_chunks), len(flowkeys)
            )
        )
        for k in flowkeys:
            if k not in flowkeys_chunkidx:
                flowkeys_chunkidx[k] = []
            flowkeys_chunkidx[k].append(chunk_id)
    return flowkeys_chunkidx


def get_global_max_flow_len(df_chunks: List[pd.DataFrame], config: dict) -> int:
    prepost_config = config.get("pre_post_processor", {}).get("config", {})
    if prepost_config["max_flow_len"]:
        return int(prepost_config["max_flow_len"])

    max_flow_lens: List[int] = []
    for chunk_id, df_chunk in enumerate(df_chunks[1:]):
        # corner case: skip for empty df_chunk
        if len(df_chunk) == 0:
            continue
        gk_chunk = df_chunk.groupby(by=[m.column for m in prepost_config["metadata"]])
        max_flow_lens.append(max(gk_chunk.size().values))

    return max(max_flow_lens)


def get_word2vec_model(
    df: pd.DataFrame, config: dict, model_directory: str
) -> Optional[Word2Vec]:
    prepost_config = config.get("pre_post_processor", {}).get("config", {})
    word2vec_cols = [
        m
        for m in (prepost_config["metadata"] + prepost_config["timeseries"])
        if "word2vec" in getattr(m, "encoding", "")
    ]

    if not word2vec_cols:
        logger.info("Skipping word2vec embedding: no word2vec columns")
        return None

    if prepost_config["word2vec"]["pretrain_model_path"]:
        logger.info("Word2vec: Loading pretrained model")
        return Word2Vec.load(prepost_config["word2vec"]["pretrain_model_path"])

    logger.info("Word2vec: training model")
    word2vec_model_path = word2vec_train(
        df=df,
        out_dir=model_directory,
        model_name=prepost_config["word2vec"]["model_name"],
        word2vec_cols=word2vec_cols,
        word2vec_size=prepost_config["word2vec"]["vec_size"],
        annoy_n_trees=prepost_config["word2vec"]["annoy_n_trees"],
    )
    return Word2Vec.load(word2vec_model_path)


def build_fields(config: dict, df: pd.DataFrame) -> Tuple[List[Field], List[Field]]:
    prepost_config = config.get("pre_post_processor", {}).get("config", {})

    metadata_fields: List[Field] = []
    timeseries_fields: List[Field] = []

    for i, field in enumerate(prepost_config["metadata"] + prepost_config["timeseries"]):
        if not isinstance(field.column, str):
            raise ValueError('"column" should be a string')
        if "type" not in field or field.type not in config["global_config"]["allowed_data_types"]:
            raise ValueError(
                '"type" must be specified as ({})'.format(
                    " | ".join(config["global_config"]["allowed_data_types"])
                )
            )

        field_name = getattr(field, "name", field.column)

        # Bit Field: (integer)
        field_instance: Field
        if "bit" in getattr(field, "encoding", ""):
            if field.type != "integer":
                raise ValueError('"encoding=bit" can be only used for "type=integer"')
            if "n_bits" not in field:
                raise ValueError("`n_bits` needs to be specified for bit fields")
            field_instance = BitField(
                name=getattr(field, "name", field.column), num_bits=field.n_bits
            )
            # applied_df = df.apply(lambda row: field_instance.normalize(
            # row[field_name]), axis='columns', result_type='expand')
            # print("applied_df:", applied_df.shape)

        # word2vec field: (any)
        if "word2vec" in getattr(field, "encoding", ""):
            field_instance = ContinuousField(
                name=field_name,
                norm_option=Normalization.MINUSONE_ONE,  # l2-norm
                dim_x=prepost_config["word2vec"]["vec_size"],
            )

        # Categorical field: (string | integer)
        if "categorical" in getattr(field, "encoding", ""):
            if field.type not in ["string", "integer"]:
                raise ValueError(
                    '"encoding=cateogrical" can be only used for "type=(string | integer)"'
                )
            field_instance = DiscreteField(
                choices=list(set(df[field.column])),
                name=getattr(field, "name", field.column),
            )

        # Continuous Field: (float)
        if field.type == "float":
            field_instance = ContinuousField(
                name=field_name,
                norm_option=getattr(Normalization, field.normalization),
                min_x=min(df[field.column]) - EPS,
                max_x=max(df[field.column]) + EPS,
                dim_x=1,
            )
            if getattr(field, "log1p_norm", False):
                df[field.column] = np.log1p(df[field.column])

        if field in prepost_config["metadata"]:
            metadata_fields.append(field_instance)
        if field in prepost_config["timeseries"]:
            timeseries_fields.append(field_instance)

    return metadata_fields, timeseries_fields


def prepare_cross_chunks_data(
    df_chunks: List[pd.DataFrame], config: dict, target_dir: str
) -> CrossChunkData:
    """
    This function splits the input data into chunks, and compute the .
    """
    big_df = pd.concat(df_chunks, axis=0, ignore_index=True)
    embed_model = get_word2vec_model(big_df, config, target_dir)
    metadata_fields, timeseries_fields = build_fields(config, big_df)
    flowkeys_chunkidx = get_flowkeys_chunkidx(df_chunks, config)
    global_max_flow_len = get_global_max_flow_len(df_chunks, config)

    return CrossChunkData(
        embed_model=embed_model,
        flowkeys_chunkidx=flowkeys_chunkidx,
        global_max_flow_len=global_max_flow_len,
        metadata_fields=metadata_fields,
        timeseries_fields=timeseries_fields,
    )
