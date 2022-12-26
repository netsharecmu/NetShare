import os
import shutil

from netshare.configs import get_config
from netshare.logger import logger


def get_pre_processed_data_folder() -> str:
    result = os.path.join(get_config()["work_folder"], "pre_processed_data")
    check_folder(result)
    return result


def get_post_processed_data_folder() -> str:
    result = os.path.join(get_config()["work_folder"], "post_processed_data")
    check_folder(result)
    return result


def get_generated_data_folder() -> str:
    result = os.path.join(get_config()["work_folder"], "generated_data")
    check_folder(result)
    return result


def get_model_folder() -> str:
    result = os.path.join(get_config()["work_folder"], "models")
    check_folder(result)
    return result


def get_visualization_folder() -> str:
    result = os.path.join(get_config()["work_folder"], "visulization")
    check_folder(result)
    return result


def get_generated_data_log_folder() -> str:
    result = os.path.join(get_config()["work_folder"], "logs", "generated_data")
    check_folder(result)
    return result


def get_model_log_folder() -> str:
    result = os.path.join(get_config()["work_folder"], "logs", "models")
    check_folder(result)
    return result


def check_folder(folder: str, skip_existing: bool = False) -> None:
    if os.path.exists(folder):
        if skip_existing:
            return
        if get_config()["global_config"]["overwrite"]:
            logger.warning(f"{folder} already exists. You are overwriting the results.")
        else:
            raise Exception(
                f"{folder} already exists. Either change the work_folder of use the overwrite option."
            )
    else:
        os.makedirs(folder, exist_ok=True)


def copy_files(source_dir: str, target_dir: str) -> None:
    """
    This function copies all files from source_dir to target_dir.
    """
    for file_name in os.listdir(source_dir):
        shutil.copy(os.path.join(source_dir, file_name), target_dir)
