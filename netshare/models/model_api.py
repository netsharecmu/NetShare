import os
from typing import Generator, Tuple

import numpy as np

from netshare.utils.paths import get_generated_data_folder


def get_generated_data() -> Generator[
    Tuple[np.ndarray, np.ndarray, np.ndarray, str], None, None
]:
    sub_folders = os.listdir(get_generated_data_folder())
    for sub_folder in sub_folders:
        for file in os.listdir(os.path.join(get_generated_data_folder(), sub_folder)):
            if file.endswith(".npz"):
                data_path = os.path.join(get_generated_data_folder(), sub_folder, file)
                data = np.load(data_path)
                unnormalized_timeseries = data["data_feature"]
                unnormalized_metadata = data["data_attribute"]
                data_gen_flag = data["data_gen_flag"]
                yield unnormalized_timeseries, unnormalized_metadata, data_gen_flag, sub_folder
