import math
import os
from typing import List, Tuple

import numpy as np
import pandas as pd

from netshare.configs import get_config
from netshare.utils.constants import EPS


def split_dataframe_to_chunks(
    big_raw_df: pd.DataFrame,
    split_type: str = "fixed_size",
    n_chunks: int = 10,
    eps: float = EPS,
) -> List[pd.DataFrame]:
    config_timestamp = get_config(
        ["pre_post_processor.config.timestamp", "global_config.timestamp"]
    )
    if not isinstance(config_timestamp, dict):
        raise ValueError(
            "The timestamp configuration is not a dictionary, please upgrade to the new format"
        )

    if n_chunks > 1 and not config_timestamp.get("column"):
        raise ValueError(
            "Trying to split into multiple chunks by timestamp but no timestamp column is provided!"
        )

    time_col_name = config_timestamp["column"]
    big_raw_df = big_raw_df.sort_values(time_col_name)

    if n_chunks == 1:
        return [big_raw_df]

    dfs = []
    if split_type == "fixed_size":
        dfs = np.array_split(big_raw_df, n_chunks)
        return dfs

    elif split_type == "fixed_time":
        time_evenly_spaced = np.linspace(
            big_raw_df[time_col_name].min(),
            big_raw_df[time_col_name].max(),
            num=n_chunks + 1,
        )
        time_evenly_spaced[-1] *= 1 + eps

        for chunk_id in range(n_chunks):
            df_chunk = big_raw_df[
                (big_raw_df[time_col_name] >= time_evenly_spaced[chunk_id])
                & (big_raw_df[time_col_name] < time_evenly_spaced[chunk_id + 1])
            ]
            if len(df_chunk) == 0:
                print("Raw chunk_id: {}, empty df_chunk!".format(chunk_id))
                continue
            dfs.append(df_chunk)
        return dfs

    else:
        raise ValueError("Unknown split type")


def load_dataframe_chunks(csv_dir: str) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
    """
    This function load the CSV files into a pandas dataframe.
    """
    dfs = []
    for filename in os.listdir(csv_dir):
        df = pd.read_csv(os.path.join(csv_dir, filename), index_col=None, header=0)
        df["filename"] = filename
        dfs.append(df)
    df = pd.concat(dfs, axis=0, ignore_index=True)
    if get_config("learn.dropna", default_value=True):
        df.dropna(inplace=True)
    if get_config("global_config.n_chunks", default_value=1) > 1:
        df_chunks = split_dataframe_to_chunks(
            big_raw_df=df,
            split_type=get_config(
                ["pre_post_processor.config.df2chunks", "global_config.df2chunks"],
                default_value=None,
            )
            or get_config("input_adapters.chunk_split_type"),
            n_chunks=get_config("global_config.n_chunks"),
        )
    else:
        df_chunks = [df]
    return df, df_chunks
