from netshare.input_adapters.input_adapter_api import put_canonical_data_from_directory
from netshare.input_adapters.normalize_format_to_canonical.base_format_normalizer import (
    DataFormatNormalizer,
)


class CsvNormalizer(DataFormatNormalizer):
    """
    This is the default normalizer, where the data is already in a csv format.
    This normalizer just move the given files to the api.
    """

    def normalize_data(self, input_dir: str) -> None:
        put_canonical_data_from_directory(input_dir)
