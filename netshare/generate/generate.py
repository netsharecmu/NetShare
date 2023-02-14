from netshare.generate.denormalize_fields import denormalize_fields
from netshare.generate.generate_api import get_generated_data_dir
from netshare.generate.model_generate import model_generate
from netshare.utils.paths import copy_files


def generate() -> None:
    """
    This is the main function of the postprocess phase.
    We get the generated data, prepare it to be exported, export it, and create the visualization.

    This function execute the following steps:
    1. Denormalize the fields (e.g. int to IP, vector to word, etc.)
    2. Choose the best generated data and write it using the generate_api
    """
    model_generate()
    denormalized_fields_dir = denormalize_fields()
    chosen_data_dir = choose_best_model(denormalized_fields_dir)
    copy_files(chosen_data_dir, get_generated_data_dir())


def choose_best_model(denormalized_fields_dir: str) -> str:
    """
    :return: the path to the data that was created by the chosen model
        (the raw data that should be shared with the user).
    """
    return denormalized_fields_dir
