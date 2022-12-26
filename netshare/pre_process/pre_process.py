import os
import tempfile
from typing import List, Tuple

import pandas as pd

from netshare import ray
from netshare.configs import get_config
from netshare.logger import logger
from netshare.pre_process.data_source import fetch_data
from netshare.pre_process.preprocess_per_chunk import preprocess_per_chunk
from netshare.pre_process.normalize_format_to_csv import normalize_files_format
from netshare.pre_process.prepare_cross_chunks_data import (
    CrossChunksData,
    prepare_cross_chunks_data,
)
from netshare.pre_post_processors.netshare.preprocess_helper import df2chunks


def pre_process() -> None:
    """
    This is the main function of the preprocess phase.
    We get the configuration, and prepare everything for the training phase.

    This function execute the following steps:
    1. Copy the data from the data source to local (e.g. from S3 bucket, DB, etc.)
    2. Normalize the files format to CSV (e.g. from pcap, json, etc.)
    3. Split the CSV data into chunks
    4. Prepare cross-chunks data:
      4.1. Word2Vec model
      4.2. The features and attributes fields
      4.3. Max flow size
      4.4. Mapping between flow keys and chunk indexes
    5. Normalize the fields in each chunk (e.g. apply Word2Vec, etc.)
    6. Use preprocess.api to save the prepared data
    """
    raw_data_dir = fetch_data()
    normalized_csv_dir = normalize_files_format(raw_data_dir)
    df, df_chunks = load_dataframe_chunks(normalized_csv_dir)
    cross_chunks_data = prepare_cross_chunks_data(df, df_chunks)
    apply_distributed_chunk_logic(df_chunks, cross_chunks_data)


def load_dataframe_chunks(csv_dir: str) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
    """
    This function load the CSV files into a pandas dataframe.
    """
    df = pd.concat(
        [
            pd.read_csv(os.path.join(csv_dir, filename), index_col=None, header=0)
            for filename in os.listdir(csv_dir)
        ],
        axis=0,
        ignore_index=True,
    )
    df.dropna(inplace=True)
    if get_config("global_config.n_chunks", default_value=1) > 1:
        df_chunks, _ = df2chunks(
            big_raw_df=df,
            config_timestamp=get_config("pre_post_processor.config.timestamp"),
            split_type=get_config("pre_post_processor.config.df2chunks"),
            n_chunks=get_config("global_config.n_chunks"),
        )
    else:
        df_chunks = [df]
    return df, df_chunks


def apply_distributed_chunk_logic(
    df_chunks: List[pd.DataFrame],
    cross_chunks_data: CrossChunksData,
) -> None:
    logger.info("Waiting for all chunks to be processed...")
    ray.get(
        [
            preprocess_per_chunk.remote(
                config=get_config(),
                cross_chunks_data=cross_chunks_data,
                df_per_chunk=df_chunk.copy(),
                chunk_id=chunk_id,
            )
            for chunk_id, df_chunk in enumerate(df_chunks)
            if len(df_chunk) != 0
        ]
    )
