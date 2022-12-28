from netshare.postprocess.denormalize_fields import denormalize_fields
from netshare.utils.paths import copy_files, get_postprocessed_data_folder


def postprocess() -> None:
    """
    This is the main function of the postprocess phase.
    We get the generated data, prepare it to be exported, export it, and create the visualization.

    This function execute the following steps:
    1. Denormalize the fields (e.g. int to IP, vector to word, etc.)
    2. Choose the best generated data
    3. Denormalize format (e.g. CSV to pcap, etc.)
    4. Export the data to the data destination (e.g. to S3 bucket, specific directory, etc.)
    """
    denormalized_fields_dir = denormalize_fields()
    chosen_data_dir = choose_best_chunk(denormalized_fields_dir)
    denormalized_format_dir = denormalize_format(chosen_data_dir)
    export_data(denormalized_format_dir)


def choose_best_chunk(denormalized_fields_dir: str) -> str:
    """
    :return: the path to the chosen data (the raw data that should be shared with the user).
    """
    return denormalized_fields_dir


def denormalize_format(denormalized_fields_dir: str) -> str:
    """
    :return: the path to the files that should be shared with the users.
    """
    return denormalized_fields_dir


def export_data(denormalized_fields_dir: str) -> None:
    """
    :return: the path to the files that should be shared with the users.
    """
    copy_files(denormalized_fields_dir, get_postprocessed_data_folder())
