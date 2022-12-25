from typing import Optional

from config_io import Config

import netshare.model_managers as model_managers
import netshare.models as models
from netshare.configs import get_config, change_work_folder, load_from_file
from netshare.post_process.post_process import post_process
from netshare.pre_process.pre_process import pre_process
from netshare.dashboard.visualize import visualize
from netshare.utils.working_directories import (
    get_pre_processed_data_folder,
    get_model_folder,
    get_generated_data_folder,
    get_model_log_folder,
    get_post_processed_data_folder,
    get_visualization_folder,
    get_generated_data_log_folder,
)


class GeneratorV2(object):
    def __init__(self, config: str):
        load_from_file(config)

        model_manager_class = getattr(model_managers, get_config("model_manager.class"))
        model_manager_config = Config(get_config("global_config"))
        model_manager_config.update(get_config("model_manager.config"))
        self._model_manager = model_manager_class(config=model_manager_config)

        model_class = getattr(models, get_config("model.class"))
        model_config = get_config("model.config")
        self._model = model_class
        self._model_config = model_config

    def generate(self, work_folder: Optional[str] = None) -> None:
        change_work_folder(work_folder)
        self._model_manager.generate(
            input_train_data_folder=get_pre_processed_data_folder(),
            input_model_folder=get_model_folder(),
            output_syn_data_folder=get_generated_data_folder(),
            log_folder=get_generated_data_log_folder(),
            create_new_model=self._model,
            model_config=self._model_config,
        )
        post_process(get_generated_data_folder())

    def train(self, work_folder: Optional[str] = None) -> None:
        change_work_folder(work_folder)
        pre_process(target_dir=get_pre_processed_data_folder())
        self._model_manager.train(
            input_train_data_folder=get_pre_processed_data_folder(),
            output_model_folder=get_model_folder(),
            log_folder=get_model_log_folder(),
            create_new_model=self._model,
            model_config=self._model_config,
        )

    def train_and_generate(self, work_folder: Optional[str] = None) -> None:
        self.train(work_folder)
        self.generate(work_folder)

    def visualize(self, work_folder: Optional[str] = None) -> None:
        change_work_folder(work_folder)
        visualize(
            generated_data_dir=get_post_processed_data_folder(),
            target_dir=get_visualization_folder(),
            original_data_dir=get_pre_processed_data_folder(),
        )
