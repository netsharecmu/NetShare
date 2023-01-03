import os
import re
from typing import Any, List, Tuple

from config_io import Config

from netshare.configs import get_config
from netshare.preprocess import preprocess_api
from netshare.utils.logger import logger
from netshare.utils.paths import get_generated_data_folder, get_model_folder


def validate_config_format(config_: dict) -> None:
    if config_.get("pretrain_non_dp") and (
        config_.get("dp_noise_multiplier") or config_.get("dp_l2_norm_clip")
    ):
        raise ValueError("pretrain_non_DP can only be used for non-DP case!")

    if (
        config_.get("pretrain_non_dp")
        and config_.get("pretrain_non_dp_reduce_time") is None
    ):
        raise ValueError(
            "pretrain_non_dp=True, then pretrain_non_dp_reduce_time must be set!"
        )

    if (
        not config_.get("pretrain_non_dp")
        and config_.get("pretrain_non_dp_reduce_time") is not None
    ):
        raise ValueError(
            "pretrain_non_dp=False, pretrain_non_dp_reduce_time does not need to be set!"
        )

    if config_.get("pretrain_non_dp") and config_.get("pretrain_dp"):
        raise ValueError("Only one of pretrain_non_DP and pretrain_DP can be True!")

    if config_.get("pretrain_dp") and config_.get("pretrain_dir") is None:
        raise ValueError(
            "You are using DP with pretrained public model, pretrain_dir must be set "
            "to the pretrained public model checkpoint directory!"
        )


def _load_config() -> List[Config]:
    config_pre_expand = Config(
        {
            **get_config("global_config"),
            **get_config("model_manager.config", default_value={}),
            **get_config("model.config", default_value={}),
        }
    )

    # TODO: add preprocessing logic for DoppelGANger (single-chunk?)
    config_pre_expand["dataset"] = []
    config_pre_expand["dataset_expand"] = True
    n_valid_chunks = 0
    for dataset_folder in preprocess_api.get_all_chunks_dirs():
        if os.path.exists(dataset_folder) and os.path.isdir(dataset_folder):
            config_pre_expand["dataset"].append(dataset_folder)
            n_valid_chunks += 1
        else:
            raise ValueError(f"Dataset folder {dataset_folder} does not exist")

    config_post_expand = config_pre_expand.expand()
    logger.info(f"Number of configurations after expanded: {len(config_post_expand)}")

    configs = []
    for config_ in config_post_expand:
        sub_result_folder = os.path.join(
            os.path.basename(config_["dataset"]),
            ",".join(
                "{}-{}".format(k, os.path.basename(str(v)))
                for k, v in config_.items()
                if f"{k}_expand" in config_.keys() and k != "dataset"
            ),
        )
        config_["sub_result_folder"] = sub_result_folder
        config_["result_folder"] = os.path.join(get_model_folder(), sub_result_folder)

        validate_config_format(config_)

        configs.append(config_)

    return configs


def get_configid_from_kv(configs: List[dict], k: Any, v: Any) -> int:
    for idx, config in enumerate(configs):
        if config[k] == v:
            return idx
    raise ValueError("{}: {} not found in configs!".format(k, v))


def create_chunks_configurations(
    generation_flag: bool = False,
) -> Tuple[List[Config], List[dict]]:
    """
    # convert a list of configurations to a grouped dictionary
    # for training purpose
    # key : value
    # dp : bool
    # dp_noise_multiplier: float
    # pretrain: bool
    # config_ids: list
    """
    configs = _load_config()

    config_id_list_victim = [i for i in range(len(configs))]
    config_group_list = []

    for config_id, config in enumerate(configs):
        if config_id in config_id_list_victim:
            config_group: dict = {
                "dp_noise_multiplier": config.get("dp_noise_multiplier"),
                "dp": (config.get("dp_noise_multiplier") is not None),
                "pretrain": (
                    config.get("pretrain_non_dp") or config.get("pretrain_dp")
                ),
                "config_ids": [],
            }

            num_chunks = config.get("n_chunks", 1)
            for chunk_idx in range(num_chunks):
                # TODO: this code should be re-written. We just wrap together the expansion made in _load_config
                config_id_ = get_configid_from_kv(
                    configs=configs,
                    k="result_folder",
                    v=re.sub(
                        "chunkid-[0-9]+",
                        "chunkid-{}".format(chunk_idx),
                        config["result_folder"],
                    ),
                )
                config_group["config_ids"].append(config_id_)
                config_id_list_victim.remove(config_id_)

            config_group_list.append(config_group)

    # sanity check
    assert len(config_id_list_victim) == 0
    config_ids_check = []
    for config_group in config_group_list:
        config_ids_check += config_group["config_ids"]
    assert set(config_ids_check) == set([i for i in range(len(configs))])

    # add pretrain_dir etc. to the original configs
    for config_group in config_group_list:
        if not config_group["pretrain"]:
            for config_id in config_group["config_ids"]:
                configs[config_id]["restore"] = False
                configs[config_id]["pretrain_dir"] = configs[config_id]["result_folder"]
        else:
            if not config_group["dp"]:
                for chunk_id, config_id in enumerate(config_group["config_ids"]):
                    if chunk_id == 0:
                        chunk0_idx = config_id
                        configs[config_id]["restore"] = False
                        iteration_range = list(
                            range(
                                configs[config_id]["extra_checkpoint_freq"] - 1,
                                configs[config_id]["iteration"],
                                configs[config_id]["extra_checkpoint_freq"],
                            )
                        )
                        iteration_range.reverse()

                        pretrain_dir = None
                        # use last available ckpt
                        if configs[config_id]["skip_chunk0_train"]:
                            last_iteration_found = False
                            for iteration_id in iteration_range:
                                if last_iteration_found:
                                    break
                                ckpt_dir = os.path.join(
                                    configs[config_id]["result_folder"],
                                    "checkpoint",
                                    "iteration_id-{}".format(iteration_id),
                                )
                                if os.path.exists(ckpt_dir):
                                    last_iteration_found = True

                            if not last_iteration_found:
                                raise ValueError(
                                    "Skipping chunk0 training but "
                                    "chunk0 has no available ckpt at {}! "
                                    "Please move ckpts into the "
                                    "corresponding folder.".format(
                                        configs[config_id]["result_folder"]
                                    )
                                )
                            else:
                                pretrain_dir = ckpt_dir
                        else:
                            if (
                                os.path.exists(
                                    os.path.join(
                                        configs[config_id]["result_folder"],
                                        "checkpoint",
                                    )
                                )
                                and not generation_flag
                            ):
                                raise ValueError(
                                    "Chunk0 training NOT skipped "
                                    "but ckpts already exist! "
                                    "Please change your working folder "
                                    "or clean up the ckpt folder "
                                    "to continute training from scratch."
                                )

                            pretrain_dir = os.path.join(
                                configs[config_id]["result_folder"],
                                "checkpoint",
                                "iteration_id-{}".format(
                                    configs[config_id]["iteration"] - 1
                                ),
                            )

                        configs[config_id]["pretrain_dir"] = pretrain_dir

                    else:
                        configs[config_id]["restore"] = True
                        configs[config_id]["pretrain_dir"] = pretrain_dir
                        configs[config_id]["iteration"] = int(
                            configs[config_id]["iteration"]
                            / configs[config_id]["pretrain_non_dp_reduce_time"]
                        )

            else:
                for chunk_id, config_id in enumerate(config_group["config_ids"]):
                    configs[config_id]["restore"] = True

    # add chunk_id and eval_root_folder for generation related
    if generation_flag:
        for config_group in config_group_list:
            chunk0_idx = config_group["config_ids"][0]
            eval_root_folder = os.path.join(
                get_generated_data_folder(),
                re.sub("chunkid-0", "", configs[chunk0_idx]["sub_result_folder"]).strip(
                    "/"
                ),
            )
            for chunk_id, config_id in enumerate(config_group["config_ids"]):
                configs[config_id]["chunk_id"] = chunk_id
                configs[config_id]["eval_root_folder"] = eval_root_folder

    for config in configs:
        os.makedirs(config["result_folder"], exist_ok=True)
        if generation_flag:
            os.makedirs(config["eval_root_folder"], exist_ok=True)

    return configs, config_group_list
