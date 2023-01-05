import os

from netshare.configs import get_config


def get_output_data_folder() -> str:
    return os.path.join(get_config("global_config.work_folder"), "output_data")
