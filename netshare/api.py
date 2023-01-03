from typing import Optional, Type

import netshare.models as models
import netshare.utils.ray as ray
from netshare.configs import change_work_folder, load_from_file
from netshare.dashboard.visualize import visualize
from netshare.models import Model
from netshare.models.model_managers import model_manager
from netshare.postprocess.postprocess import postprocess
from netshare.preprocess.preprocess import preprocess
from netshare.utils.paths import (
    get_postprocessed_data_folder,
    get_preprocessed_data_folder,
    get_visualization_folder,
)


class Generator(object):
    def __init__(self, config: str):
        load_from_file(config)
        ray.init(address="auto")
        self._model: Type[Model] = models.build_model_from_config()

    def train(self, work_folder: Optional[str] = None) -> None:
        change_work_folder(work_folder)
        preprocess()
        model_manager.train(create_new_model=self._model)

    def generate(self, work_folder: Optional[str] = None) -> None:
        change_work_folder(work_folder)
        model_manager.generate(create_new_model=self._model)
        postprocess()

    def train_and_generate(self, work_folder: Optional[str] = None) -> None:
        self.train(work_folder)
        self.generate(work_folder)
        ray.shutdown()

    def visualize(self, work_folder: Optional[str] = None) -> None:
        change_work_folder(work_folder)
        visualize(
            generated_data_dir=get_postprocessed_data_folder(),
            target_dir=get_visualization_folder(),
            original_data_dir=get_preprocessed_data_folder(),
        )
