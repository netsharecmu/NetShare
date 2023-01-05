from typing import List

import pandas as pd

from netshare.input_adapters.input_adapter_api import get_canonical_data_dir
from netshare.learn.model_learn import train_config_group
from netshare.learn.setup_cross_chunks_data import (
    CrossChunksData,
    setup_cross_chunks_data,
)
from netshare.learn.setup_per_chunk_data import setup_per_chunk
from netshare.learn.utils.dataframe_utils import load_dataframe_chunks
from netshare.utils import ray
from netshare.utils.logger import logger
from netshare.utils.model_configuration import create_chunks_configurations


def learn() -> None:
    """
    This is the main function of the input_adapters phase.
    We get the configuration, and prepare everything for the training phase.

    This function execute the following steps and stores the results using the learn_api:
    1. Split the CSV data into chunks
    2. Prepare cross-chunks data:
      2.1. Word2Vec model
      2.2. The features and attributes fields
      2.3. Max flow size
      2.4. Mapping between flow keys and chunk indexes
    3. Normalize the fields in each chunk (e.g. apply Word2Vec, etc.)
    """
    canonical_data_dir = get_canonical_data_dir()
    df, df_chunks = load_dataframe_chunks(canonical_data_dir)
    cross_chunks_data = setup_cross_chunks_data(df, df_chunks)
    apply_distributed_chunk_logic(df_chunks, cross_chunks_data)
    _train()


def apply_distributed_chunk_logic(
    df_chunks: List[pd.DataFrame],
    cross_chunks_data: CrossChunksData,
) -> None:
    logger.info(f"Waiting for all chunks to be preprocessed ({len(df_chunks)} chunks)")
    ray.get(
        [
            setup_per_chunk.remote(
                cross_chunks_data=cross_chunks_data,
                df_per_chunk=df_chunk.copy(),
                chunk_id=chunk_id,
            )
            for chunk_id, df_chunk in enumerate(df_chunks)
            if len(df_chunk) != 0
        ]
    )


def _train() -> None:
    configs, config_group_list = create_chunks_configurations(generation_flag=False)

    ray.get(
        [
            train_config_group.remote(
                config_group=config_group,
                configs=configs,
            )
            for config_group in config_group_list
        ]
    )
