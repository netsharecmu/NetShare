from typing import Optional

import netshare.utils.ray as ray
from netshare.configs import change_work_folder, load_from_file
from netshare.generate.generate import generate
from netshare.input_adapters.adapt_input import adapt_input
from netshare.learn.learn import learn
from netshare.output_adapters.adapt_output import adapt_output, get_output_data_folder
from netshare.output_adapters.dashboard import visualize_api
from netshare.utils.paths import get_preprocessed_data_folder, get_visualization_folder


class Generator(object):
    def __init__(self, config: str, work_folder: Optional[str] = None):
        load_from_file(config)
        change_work_folder(work_folder)
        ray.init(address="auto")

    def train(self) -> None:
        adapt_input()
        learn()

    def generate(self) -> None:
        generate()
        adapt_output()

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
