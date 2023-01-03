from typing import List

import pandas as pd

from netshare.preprocess.data_source import fetch_data
from netshare.preprocess.normalize_format_to_csv import normalize_files_format
from netshare.preprocess.prepare_cross_chunks_data import (
    CrossChunksData,
    prepare_cross_chunks_data,
)
from netshare.preprocess.preprocess_per_chunk import preprocess_per_chunk
from netshare.preprocess.utils.dataframe_utils import load_dataframe_chunks
from netshare.utils import ray
from netshare.utils.logger import logger


def preprocess() -> None:
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


def apply_distributed_chunk_logic(
    df_chunks: List[pd.DataFrame],
    cross_chunks_data: CrossChunksData,
) -> None:
    logger.info(f"Waiting for all chunks to be preprocessed ({len(df_chunks)} chunks)")
    ray.get(
        [
            preprocess_per_chunk.remote(
                cross_chunks_data=cross_chunks_data,
                df_per_chunk=df_chunk.copy(),
                chunk_id=chunk_id,
            )
            for chunk_id, df_chunk in enumerate(df_chunks)
            if len(df_chunk) != 0
        ]
    )
