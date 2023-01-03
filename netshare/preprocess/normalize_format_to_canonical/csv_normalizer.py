from netshare.preprocess.normalize_format_to_canonical.base_format_normalizer import (
    DataFormatNormalizer,
)


class CsvNormalizer(DataFormatNormalizer):
    """
    This is the default normalizer, where the data is already in a csv format.
    This normalizer just returns the same directory that it gets.
    """

    def normalize_data(self, input_dir: str) -> str:
        return input_dir
