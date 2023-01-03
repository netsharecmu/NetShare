import os
from typing import Generator, Tuple

import numpy as np

from netshare.logger import logger
from netshare.utils.paths import get_generated_data_folder


def get_generated_data() -> Generator[
    Tuple[np.ndarray, np.ndarray, np.ndarray, str], None, None
]:
    for path, directories, files in os.walk(get_generated_data_folder()):
        for directory in directories:
            if directory == "feat_raw":
                feat_raw_dir = os.path.join(path, directory)
                for file in os.listdir(feat_raw_dir):
                    if file.endswith(".npz"):
                        data = np.load(os.path.join(feat_raw_dir, file))
                        unnormalized_timeseries = data["data_feature"]
                        unnormalized_metadata = data["data_attribute"]
                        data_gen_flag = data["data_gen_flag"]
                        subdir = feat_raw_dir[
                            len(get_generated_data_folder()) + 1 : -len("feat_raw")
                        ]
                        yield unnormalized_timeseries, unnormalized_metadata, data_gen_flag, subdir
