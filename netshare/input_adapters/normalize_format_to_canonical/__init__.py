from netshare.configs import get_config
from netshare.utils.logger import logger

from .base_format_normalizer import DataFormatNormalizer
from .csv_normalizer import CsvNormalizer
from .jsons_normalizer import JsonsNormalizer
from .pcap_format_normalizer import PcapNormalizer


def normalize_files_format(input_dir: str) -> None:
    """
    This is the builder function for the files format normalizer.
    This function return the path to the directory that contains the normalized files.
    """
    normalizer_type = (
        get_config(
            [
                "dataset_type",
                "global_config.dataset_type",
                "input_adapters.format_normalizer.dataset_type",
            ],
            default_value=None,
        )
        or "csv"
    )

    format_normalizer: DataFormatNormalizer
    if normalizer_type == "pcap":
        logger.info("Normalizing files format from pcap")
        format_normalizer = PcapNormalizer()
    elif normalizer_type == "list-of-jsons":
        logger.info("Normalizing files format from list of JSONs")
        format_normalizer = JsonsNormalizer()
    elif normalizer_type == "csv":
        logger.info("Skip format normalize: files are already in csv format")
        format_normalizer = CsvNormalizer()
    else:
        raise ValueError(f"Unknown format normalizer type: {normalizer_type}")

    format_normalizer.normalize_data(input_dir)
