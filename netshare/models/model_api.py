import os
from typing import Generator, Tuple

import numpy as np

from netshare.logger import logger
from netshare.utils.paths import get_generated_data_folder


def get_generated_data() -> Generator[
    Tuple[np.ndarray, np.ndarray, np.ndarray, str], None, None
]:
    sub_folders = os.listdir(get_generated_data_folder())
    for sub_folder in sub_folders:
        data_dir = os.path.join(get_generated_data_folder(), sub_folder, "feat_raw")
        if not os.path.exists(data_dir):
            logger.info(f"Couldn't find generated data in directory {data_dir}")
            continue

        for file in os.listdir(data_dir):
            if file.endswith(".npz"):
                data = np.load(os.path.join(data_dir, file))
                unnormalized_timeseries = data["data_feature"]
                unnormalized_metadata = data["data_attribute"]
                data_gen_flag = data["data_gen_flag"]
                yield unnormalized_timeseries, unnormalized_metadata, data_gen_flag, sub_folder
