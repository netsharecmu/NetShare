from typing import Optional, Type

import netshare.models as models
from netshare.configs import get_config, change_work_folder, load_from_file
from netshare.model_managers import ModelManager, build_model_manager_from_config
from netshare.models import Model
from netshare.post_process.post_process import post_process
from netshare.pre_process.pre_process import pre_process
from netshare.dashboard.visualize import visualize
from netshare.utils.paths import (
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
        self._model_manager: ModelManager = build_model_manager_from_config()
        self._model: Type[Model] = models.build_model_from_config()

    def generate(self, work_folder: Optional[str] = None) -> None:
        change_work_folder(work_folder)
        self._model_manager.generate(
            input_train_data_folder=get_pre_processed_data_folder(),
            input_model_folder=get_model_folder(),
            output_syn_data_folder=get_generated_data_folder(),
            log_folder=get_generated_data_log_folder(),
            create_new_model=self._model,
            model_config=get_config("model.config"),
        )
        post_process(get_generated_data_folder())

    def train(self, work_folder: Optional[str] = None) -> None:
        change_work_folder(work_folder)
        pre_process()
        self._model_manager.train(
            input_train_data_folder=get_pre_processed_data_folder(),
            output_model_folder=get_model_folder(),
            log_folder=get_model_log_folder(),
            create_new_model=self._model,
            model_config=get_config("model.config"),
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
