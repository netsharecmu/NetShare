import os
from typing import List

import netshare.utils.ray as ray
from netshare import models
from netshare.utils.logger import logger
from netshare.utils.paths import (
    get_generated_data_folder,
    get_model_log_folder,
    get_preprocessed_data_folder,
)


@ray.remote(scheduling_strategy="SPREAD", max_calls=1)
def train_model(
    training_configuration: dict,
) -> None:
    if not os.path.exists(training_configuration["pretrain_dir"]):
        os.makedirs(training_configuration["pretrain_dir"], exist_ok=True)
    model = models.build_model_from_config()(training_configuration)
    model.train(
        input_train_data_folder=get_preprocessed_data_folder(),
        output_model_folder=get_generated_data_folder(),
        log_folder=get_model_log_folder(),
    )


def should_train_chunk_0(config_group: dict, chunk0_config: dict) -> bool:
    if not config_group.get("dp", True) and config_group.get("pretrain", False):
        if chunk0_config.get("skip_chunk0_train", False):
            if not chunk0_config.get("pretrain_dir"):
                raise ValueError(
                    "skip_chunk0_train marked as true in the configuration, but pretrain_dir was not provided."
                )
            return False
    return True


@ray.remote(scheduling_strategy="SPREAD")
def train_config_group(
    config_group: dict,
    configs: List[dict],
) -> None:
    """
    This function train the given config group and list of configs.
    With no differential privacy, we would first train a model given the first chunk of data (D_chunk0).
        Then the rest of the models will be trained by using the first trained model as a base and fine-tuning
        the parameters(weights) given the different chunks of data(D_chunk1 - D_chunkN).
    With differential privacy <Note: not implemented yet>, we would first train a pre-model using a public dataset.
        Then the N+1 models will be trained by fine-tuning the pre-model given data(D_chunk0 - D_chunkN)
        while DP-SGD is being activated.
    """
    chunk0_config = configs[config_group["config_ids"][0]]
    if should_train_chunk_0(config_group, chunk0_config):
        logger.info("Start training of chunk0")
        ray.get(train_model.remote(chunk0_config))
    else:
        logger.info("Skipping chunk0 training")

    logger.info("Start training the rest of the chunks")
    ray.get(
        [
            train_model.remote(configs[config_idx])
            for config_idx in config_group["config_ids"][1:]
        ]
    )
