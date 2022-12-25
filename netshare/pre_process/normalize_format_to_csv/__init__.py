from netshare.logger import logger
from netshare.configs import get_config

from .default_format_normalizer import CsvNormalizer
from .pcap_format_normalizer import PcapNormalizer
from .base_format_normalizer import DataFormatNormalizer


def normalize_files_format(input_dir: str, target_dir: str) -> None:
    """
    This is the builder function for the files format normalizer.
    """
    format_normalizer_config = (
        get_config().get("pre_process", {}).get("format_normalizer", {})
    )
    normalizer_type = (
        get_config().get("dataset_type")
        or format_normalizer_config.get("dataset_type")
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

    format_normalizer.normalize_data(input_dir, target_dir)
