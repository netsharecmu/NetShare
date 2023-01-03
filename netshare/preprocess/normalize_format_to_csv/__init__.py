from netshare.configs import get_config
from netshare.utils.logger import logger

from .base_format_normalizer import DataFormatNormalizer
from .default_format_normalizer import CsvNormalizer
from .pcap_format_normalizer import PcapNormalizer


def normalize_files_format(input_dir: str) -> str:
    """
    This is the builder function for the files format normalizer.
    This function return the path to the directory that contains the normalized files.
    """
    format_normalizer_config = get_config(
        "preprocess.format_normalizer", default_value={}
    )
    normalizer_type = (
        get_config(
            "dataset_type", path2="global_config.dataset_type", default_value=None
        )
        or format_normalizer_config.get("dataset_type", None)
        or "csv"
    )

    format_normalizer: DataFormatNormalizer
    if normalizer_type == "pcap":
        logger.info("Normalizing files format from pcap")
        format_normalizer = PcapNormalizer()
    elif normalizer_type == "csv":
        logger.info("Skip format normalize: files are already in csv format")
        format_normalizer = CsvNormalizer()
    else:
        raise ValueError(f"Unknown format normalizer type: {normalizer_type}")

    return format_normalizer.normalize_data(input_dir)
