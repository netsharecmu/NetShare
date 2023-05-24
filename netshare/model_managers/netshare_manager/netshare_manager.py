import inspect

from ..model_manager import ModelManager
from .train_helper import _train_specific_config_group
from .generate_helper import _generate_attr, _merge_attr, _generate_given_attr, _generate_session
from .netshare_util import _load_config, _configs2configsgroup
import netshare.ray as ray
import os
import time
import json

import pandas as pd


class NetShareManager(ModelManager):
    def _train(self, input_train_data_folder, output_model_folder, log_folder,
               create_new_model, model_config):
        print(f"{self.__class__.__name__}.{inspect.stack()[0][3]}")

        configs = _load_config(
            config_dict={
                **self._config,
                **model_config},
            input_train_data_folder=input_train_data_folder,
            output_model_folder=output_model_folder)

        configs, config_group_list = _configs2configsgroup(
            configs=configs,
            generation_flag=False)
        print(config_group_list)
        with open(os.path.join(output_model_folder, "configs_train.json"), 'w') as f:
            json.dump({
                "configs": configs,
                "config_group_list": config_group_list
            }, f, indent=4)

        objs = []
        for config_group_id, config_group in enumerate(config_group_list):
            objs.append(
                _train_specific_config_group.remote(
                    create_new_model=create_new_model,
                    config_group_id=config_group_id,
                    config_group=config_group,
                    configs=configs,
                    input_train_data_folder=input_train_data_folder,
                    output_model_folder=output_model_folder,
                    log_folder=log_folder)
            )
        results = ray.get(objs)
        return results

    def _generate(
            self, input_train_data_folder, input_model_folder,
            output_syn_data_folder, log_folder, create_new_model, model_config):
        configs = _load_config(
            config_dict={
                **self._config,
                **model_config},
            input_train_data_folder=input_train_data_folder,
            output_model_folder=input_model_folder)

        configs, config_group_list = _configs2configsgroup(
            configs=configs,
            generation_flag=True,
            output_syn_data_folder=output_syn_data_folder
        )

        with open(os.path.join(output_syn_data_folder, "configs_generate.json"), 'w') as f:
            json.dump({
                "configs": configs,
                "config_group_list": config_group_list
            }, f, indent=4)

        print("Start generating attributes ...")
        if configs[0]["n_chunks"] > 1:
            objs = []
            for config_idx, config in enumerate(configs):
                objs.append(
                    _generate_attr.remote(
                        create_new_model=create_new_model,
                        configs=configs,
                        config_idx=config_idx,
                        log_folder=log_folder))
            _ = ray.get(objs)
            time.sleep(10)
            print("Finish generating attributes")

            print("Start merging attributes ...")
            objs = []
            for config_group in config_group_list:
                chunk0_idx = config_group["config_ids"][0]
                eval_root_folder = configs[chunk0_idx]["eval_root_folder"]

                objs.append(
                    _merge_attr.remote(
                        attr_raw_npz_folder=os.path.join(
                            eval_root_folder, "attr_raw"),
                        config_group=config_group,
                        configs=configs)
                )
            _ = ray.get(objs)
            time.sleep(10)
            print("Finish merging attributes...")

            print("Start generating features given attributes ...")
            objs = []
            for config_idx, config in enumerate(configs):
                objs.append(
                    _generate_given_attr.remote(
                        create_new_model=create_new_model,
                        configs=configs,
                        config_idx=config_idx,
                        log_folder=log_folder))
            _ = ray.get(objs)
            time.sleep(10)
        else:
            objs = []
            for config_idx, config in enumerate(configs):
                objs.append(
                    _generate_session.remote(
                        create_new_model=create_new_model,
                        configs=configs,
                        config_idx=config_idx,
                        log_folder=log_folder))
            _ = ray.get(objs)
        print("Finish generating features given attributes ...")

        return True
