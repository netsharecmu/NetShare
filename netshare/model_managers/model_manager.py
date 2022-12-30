import time
from typing import Type

import netshare.ray as ray
from netshare.logger import logger
from netshare.model_managers.netshare_manager.generate_helper import (
    generate_data,
    merge_attr,
)
from netshare.model_managers.netshare_manager.netshare_util import (
    create_chunks_configurations,
)
from netshare.model_managers.netshare_manager.train_helper import train_config_group
from netshare.models import Model


def train(create_new_model: Type[Model]) -> None:
    configs, config_group_list = create_chunks_configurations(generation_flag=False)

    ray.get(
        [
            train_config_group.remote(
                create_new_model=create_new_model,
                config_group=config_group,
                configs=configs,
            )
            for config_group in config_group_list
        ]
    )


def generate(create_new_model: Type[Model]) -> None:
    configs, config_group_list = create_chunks_configurations(generation_flag=True)

    logger.info("Start generating attributes")
    ray.get(
        [
            generate_data.remote(
                create_new_model=create_new_model,
                config=config,
                given_data_attribute_flag=False,
            )
            for config in configs
        ]
    )
    time.sleep(10)

    # TODO: The below part didn't happen in dg. Why?
    logger.info("Start merging attributes")
    ray.get(
        [
            merge_attr.remote(config_group=config_group, configs=configs)
            for config_group in config_group_list
        ]
    )
    time.sleep(10)

    logger.info("Start generating features given attributes")
    ray.get(
        [
            generate_data.remote(
                create_new_model=create_new_model,
                config=config,
                given_data_attribute_flag=True,
            )
            for config in configs
        ]
    )
    time.sleep(10)
