import os

from netshare.configs import get_config
from netshare.logger import logger


def get_pre_processed_data_folder() -> str:
    result = os.path.join(get_config()["work_folder"], "pre_processed_data")
    _check_folder(result)
    return result


def get_post_processed_data_folder() -> str:
    result = os.path.join(get_config()["work_folder"], "post_processed_data")
    _check_folder(result)
    return result


def get_generated_data_folder() -> str:
    result = os.path.join(get_config()["work_folder"], "generated_data")
    _check_folder(result)
    return result


def get_model_folder() -> str:
    result = os.path.join(get_config()["work_folder"], "models")
    _check_folder(result)
    return result


def get_visualization_folder() -> str:
    result = os.path.join(get_config()["work_folder"], "visulization")
    _check_folder(result)
    return result


def get_generated_data_log_folder() -> str:
    result = os.path.join(get_config()["work_folder"], "logs", "generated_data")
    _check_folder(result)
    return result


def get_model_log_folder() -> str:
    result = os.path.join(get_config()["work_folder"], "logs", "models")
    _check_folder(result)
    return result


def _check_folder(folder: str) -> None:
    if os.path.exists(folder):
        if get_config()["global_config"]["overwrite"]:
            logger.warning(f"{folder} already exists. You are overwriting the results.")
        else:
            raise Exception(
                f"{folder} already exists. Either change the work_folder of use the overwrite option."
            )
    else:
        os.makedirs(folder, exist_ok=True)
