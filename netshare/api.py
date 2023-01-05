from typing import Optional

import netshare.utils.ray as ray
from netshare.configs import change_work_folder, load_from_file
from netshare.generate.generate import generate
from netshare.input_adapters.adapt_input import input_adapter
from netshare.learn.learn import learn
from netshare.output_adapters.dashboard import visualize_api
from netshare.output_adapters.output_adapter import (
    get_output_data_folder,
    output_adapter,
)
from netshare.utils.paths import get_preprocessed_data_folder, get_visualization_folder


class Generator(object):
    def __init__(self, config: str, work_folder: Optional[str] = None):
        load_from_file(config)
        change_work_folder(work_folder)
        ray.init(address="auto")

    def train(self) -> None:
        input_adapter()
        learn()

    def generate(self) -> None:
        generate()
        output_adapter()

    def train_and_generate(self) -> None:
        self.train()
        self.generate()
        ray.shutdown()

    def visualize(self, work_folder: Optional[str] = None) -> None:
        change_work_folder(work_folder)
        visualize_api.visualize(  # type: ignore
            generated_data_dir=get_output_data_folder(),
            target_dir=get_visualization_folder(),
            original_data_dir=get_preprocessed_data_folder(),
        )
