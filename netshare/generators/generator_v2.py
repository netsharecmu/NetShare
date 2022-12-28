from typing import Optional, Type

import netshare.models as models
from netshare.configs import change_work_folder, get_config, load_from_file
from netshare.dashboard.visualize import visualize
from netshare.model_managers import ModelManager, build_model_manager_from_config
from netshare.models import Model
from netshare.postprocess.postprocess import postprocess
from netshare.preprocess.preprocess import preprocess
from netshare.utils.paths import (
    get_generated_data_folder,
    get_generated_data_log_folder,
    get_model_folder,
    get_model_log_folder,
    get_postprocessed_data_folder,
    get_preprocessed_data_folder,
    get_visualization_folder,
)


class GeneratorV2(object):
    def __init__(self, config: str):
        load_from_file(config)
        self._model_manager: ModelManager = build_model_manager_from_config()
        self._model: Type[Model] = models.build_model_from_config()

    def generate(self, work_folder: Optional[str] = None) -> None:
        change_work_folder(work_folder)
        self._model_manager.generate(
            input_train_data_folder=get_preprocessed_data_folder(),
            input_model_folder=get_model_folder(),
            output_syn_data_folder=get_generated_data_folder(),
            log_folder=get_generated_data_log_folder(),
            create_new_model=self._model,
            model_config=get_config("model.config"),
        )
        postprocess()

    def train(self, work_folder: Optional[str] = None) -> None:
        change_work_folder(work_folder)
        preprocess()
        self._model_manager.train(
            input_train_data_folder=get_preprocessed_data_folder(),
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
            generated_data_dir=get_postprocessed_data_folder(),
            target_dir=get_visualization_folder(),
            original_data_dir=get_preprocessed_data_folder(),
        )
