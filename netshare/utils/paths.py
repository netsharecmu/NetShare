import os
import shutil

from netshare.configs import get_config
from netshare.utils.logger import logger


def get_preprocessed_data_folder() -> str:
    return os.path.join(get_config("global_config.work_folder"), "pre_processed_data")


def get_generated_data_folder() -> str:
    return os.path.join(get_config("global_config.work_folder"), "generated_data")


def get_model_folder() -> str:
    result = os.path.join(get_config("global_config.work_folder"), "models")
    check_folder(result, skip_existing=True)
    return result


def get_visualization_folder() -> str:
    return os.path.join(get_config("global_config.work_folder"), "visulization")


def get_generated_data_log_folder() -> str:
    return os.path.join(
        get_config("global_config.work_folder"), "logs", "generated_data"
    )


def get_model_log_folder() -> str:
    return os.path.join(get_config("global_config.work_folder"), "logs", "models")


def get_word2vec_model_directory() -> str:
    return get_preprocessed_data_folder()


def get_word2vec_model_path() -> str:
    word2vec_config = get_config(
        ["pre_post_processor.config", "learn"], default_value={}
    )
    word2vec_model_path = os.path.join(
        get_word2vec_model_directory(),
        "{}_{}.model".format(
            word2vec_config["word2vec"]["model_name"],
            word2vec_config["word2vec"]["vec_size"],
        ),
    )
    return word2vec_model_path


def get_annoyIndex_for_word2vec(name: str) -> str:
    annoyIndex_path = os.path.join(
        get_word2vec_model_directory(), "{}.ann".format(name)
    )
    return annoyIndex_path


def get_annoy_dict_idx_ele_for_word2vec() -> str:
    annoy_type_dict_path = os.path.join(
        get_word2vec_model_directory(), "annoy_idx_ele_dict.json"
    )
    return annoy_type_dict_path


def check_folder(folder: str, skip_existing: bool = False) -> None:
    if os.path.exists(folder):
        if skip_existing:
            return
        if get_config("global_config.overwrite", default_value=False):
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
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    shutil.copytree(source_dir, target_dir)
