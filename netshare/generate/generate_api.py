import os
from typing import Generator, Tuple

import numpy as np

from netshare.configs import get_config
from netshare.utils.paths import copy_files, get_generated_data_folder


def get_raw_generated_data_dir() -> str:
    return os.path.join(get_config("global_config.work_folder"), "raw_generated")


def get_generated_data_dir() -> str:
    return os.path.join(get_config("global_config.work_folder"), "generated")


def get_canonical_data_dir() -> str:
    """
    This function returns a directory that contains the canonical data.
    """
    return os.path.join(get_config("global_config.work_folder"), "canonical_output")


def put_canonical_data_from_directory(input_dir: str) -> None:
    """
    This function gets the files from the given directory, and put them in the canonical data store.
    It assumes that the files are in the canonical format - CSV files.
    """
    copy_files(input_dir, get_canonical_data_dir())


def get_raw_generated_data() -> Generator[
    Tuple[np.ndarray, np.ndarray, np.ndarray, str], None, None
]:
    for path, directories, files in os.walk(get_raw_generated_data_dir()):
        for directory in directories:
            if directory == "feat_raw":
                feat_raw_dir = os.path.join(path, directory)
                for file in os.listdir(feat_raw_dir):
                    if file.endswith(".npz"):
                        data = np.load(os.path.join(feat_raw_dir, file))
                        unnormalized_timeseries = data["data_feature"]
                        unnormalized_session_key = data["data_attribute"]
                        data_gen_flag = data["data_gen_flag"]
                        subdir = feat_raw_dir[
                            len(get_generated_data_folder()) : -len("feat_raw")
                        ]
                        if subdir.startswith("pre_processed_data"):
                            continue
                        yield unnormalized_timeseries, unnormalized_session_key, data_gen_flag, subdir
