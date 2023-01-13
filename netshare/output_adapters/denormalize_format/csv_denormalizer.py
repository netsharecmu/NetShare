from netshare.output_adapters.denormalize_format.base_format_denormalizer import (
    DataFormatDenormalizer,
)


class CsvDeormalizer(DataFormatDenormalizer):
    """
    This is the default denormalizer, where the output data should be in a csv format.
    """

    def denormalize_data(self, input_dir: str) -> str:
        return input_dir
