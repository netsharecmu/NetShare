import os
import shutil

from netshare.pre_process.normalize_format_to_csv.base_format_normalizer import (
    DataFormatNormalizer,
)


class CsvNormalizer(DataFormatNormalizer):
    """
    This is the default normalizer, where the data is already in a csv format.
    This normalizer just copies the data from the input_dir to the target_dir.
    """

    def normalize_data(self, input_dir: str, target_dir: str, config: dict) -> None:
        for file_name in os.listdir(input_dir):
            shutil.copy(os.path.join(input_dir, file_name), target_dir)
