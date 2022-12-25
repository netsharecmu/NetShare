import os
import shutil

from netshare.pre_process.normalize_format_to_csv.base_format_normalizer import (
    DataFormatNormalizer,
)
from netshare.utils.working_directories import copy_files


class CsvNormalizer(DataFormatNormalizer):
    """
    This is the default normalizer, where the data is already in a csv format.
    This normalizer just copies the data from the input_dir to the target_dir.
    """

    def normalize_data(self, input_dir: str, target_dir: str) -> None:
        copy_files(input_dir, target_dir)
