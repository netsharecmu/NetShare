from netshare.generate import generate_api
from netshare.output_adapters.denormalize_format import denormalize_files_format
from netshare.output_adapters.output_adapter_api import get_output_data_folder
from netshare.utils.logger import logger
from netshare.utils.paths import copy_files


def adapt_output() -> None:
    """
    This is the main function of the output adapter phase.
    We get the generated data, denormalize the canonical format, and export it.

    This function execute the following steps:
    1. Denormalize format (e.g. CSV to pcap, etc.)
    2. Export the data to the data destination (e.g. to S3 bucket, specific directory, etc.)
    """
    denormalized_format_dir = denormalize_files_format(
        generate_api.get_generated_data_dir()
    )
    export_data(denormalized_format_dir)


def export_data(denormalized_fields_dir: str) -> None:
    """
    :return: the path to the files that should be shared with the users.
    """
    copy_files(denormalized_fields_dir, get_output_data_folder())
    logger.info(f"Generated data exported to {get_output_data_folder()}")
