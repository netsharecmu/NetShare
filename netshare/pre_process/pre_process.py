import copy
import os
import tempfile
from typing import List

import pandas as pd

from netshare import ray
from netshare.logger import logger
from netshare.pre_post_processors.netshare.preprocess_helper import (
    df2chunks,
    split_per_chunk,
)
from netshare.pre_process.data_source import fetch_data
from netshare.pre_process.normalize_format_to_csv import normalize_files_format

from netshare.pre_process.prepare_cross_chunks_data import (
    prepare_cross_chunks_data,
    CrossChunkData,
)


def pre_process(config: dict, target_dir: str) -> None:
    """
    This is the main function of the preprocess phase.
    We get the configuration, and prepare everything for the training phase.

    This function execute the following steps:
    1. Copy the data from the data source to local (e.g. from S3 bucket, DB, etc.)
    2. Normalize the files format to CSV (e.g. from pcap, json, etc.)
    3. Split the CSV data into chunks
    4. Prepare cross-chunks data:
      4.1. Word2Vec model
      4.2. metadata and timeseries fields
      4.3. max flow size
      4.4. mapping between flow keys and chunk indexes
    5. Normalize the fields in each chunk (e.g. apply Word2Vec, etc.)
    6. Save each chunk to disk
    """
    raw_data_dir, normalized_csv_dir = tempfile.mkdtemp(), tempfile.mkdtemp()

    fetch_data(config, raw_data_dir)
    normalize_files_format(raw_data_dir, normalized_csv_dir, config)
    df_chunks = load_dataframe_chunks(normalized_csv_dir, config)
    cross_chunks_data = prepare_cross_chunks_data(df_chunks, config, target_dir)
    apply_distributed_chunk_logic(df_chunks, cross_chunks_data, config, target_dir)


def load_dataframe_chunks(csv_dir: str, config: dict) -> List[pd.DataFrame]:
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
    df_chunks, _ = df2chunks(
        big_raw_df=df,
        config_timestamp=config['pre_post_processor']['config']["timestamp"],
        split_type=config['pre_post_processor']['config']["df2chunks"],
        n_chunks=config['global_config']["n_chunks"],
    )
    return df_chunks  # type: ignore


def apply_distributed_chunk_logic(
    df_chunks: List[pd.DataFrame],
    cross_chunks_data: CrossChunkData,
    config: dict,
    target_dir: str,
) -> None:
    objs = []
    for chunk_id, df_chunk in enumerate(df_chunks):
        # skip empty df_chunk: corner case
        if len(df_chunk) == 0:
            print("Chunk_id {} empty! Skipping ...".format(chunk_id))
            continue

        print("\nChunk_id:", chunk_id)
        merged_config = {**config['global_config'], **config['pre_post_processor']['config']}
        objs.append(
            split_per_chunk.remote(
                config=merged_config,
                metadata_fields=copy.deepcopy(cross_chunks_data.metadata_fields),
                timeseries_fields=copy.deepcopy(cross_chunks_data.timeseries_fields),
                df_per_chunk=df_chunk.copy(),
                embed_model=cross_chunks_data.embed_model,
                global_max_flow_len=cross_chunks_data.global_max_flow_len,
                chunk_id=chunk_id,
                data_out_dir=os.path.join(target_dir, f"chunkid-{chunk_id}"),
                flowkeys_chunkidx=cross_chunks_data.flowkeys_chunkidx,
            )
        )

    logger.info("Waiting for all chunks to be processed...")
    ray.get(objs)
