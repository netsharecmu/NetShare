from netshare.pre_process.normalize_format_to_csv.base_format_normalizer import (
    DataFormatNormalizer,
)
from netshare.utils.paths import copy_files


class CsvNormalizer(DataFormatNormalizer):
    """
    This is the default normalizer, where the data is already in a csv format.
    This normalizer just returns the same directory that it gets.
    """

    def normalize_data(self, input_dir: str) -> str:
        return input_dir
