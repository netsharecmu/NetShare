import os

from netshare.configs import get_config
from netshare.utils.paths import copy_files


def put_canonical_data_from_directory(input_dir: str) -> None:
    """
    This function gets the files from the given directory, and put them in the canonical data store.
    It assumes that the files are in the canonical format - CSV files.
    """
    copy_files(input_dir, get_canonical_data_dir())


def get_canonical_data_dir() -> str:
    """
    This function returns a directory that contains the canonical data.
    """
    return os.path.join(get_config("global_config.work_folder"), "canonical_input")
