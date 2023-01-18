from netshare.configs import get_config
from netshare.utils.logger import logger

from .base_format_denormalizer import DataFormatDenormalizer
from .csv_denormalizer import CsvDeormalizer


def denormalize_files_format(input_dir: str) -> str:
    """
    This is the builder function for the files format denormalizer.
    This function return the path to the directory that contains the denormalized files.
    """
    normalizer_type = get_config(
        "output_adapters.format_denormalizer.dataset_type", default_value="csv"
    )

    format_normalizer: DataFormatDenormalizer
    if normalizer_type == "csv":
        logger.info("Skip format denormalize: files should be in csv format")
        format_normalizer = CsvDeormalizer()
    else:
        raise ValueError(f"Unknown format normalizer type: {normalizer_type}")

    return format_normalizer.denormalize_data(input_dir)
