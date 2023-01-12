import csv
import json
import os
from typing import Union

from netshare.configs import get_config
from netshare.input_adapters.input_adapter_api import get_canonical_data_dir
from netshare.input_adapters.normalize_format_to_canonical.base_format_normalizer import (
    DataFormatNormalizer,
)


class JsonsNormalizer(DataFormatNormalizer):
    """
    This format normalizer reads data files that are jsons, one json per line,
        and converts the given values in the given keys to canonical format.
    """

    @staticmethod
    def extract_column(
        single_row_json: dict, key_configuration: str
    ) -> Union[str, int]:
        if "." not in key_configuration:
            result = single_row_json[key_configuration]
            if not isinstance(result, (int, str)):
                raise ValueError(
                    f"Value of key {key_configuration} is not a string or an int"
                )
            return result
        current_key, other_keys = key_configuration.split(".", maxsplit=1)
        return JsonsNormalizer.extract_column(single_row_json[current_key], other_keys)

    def normalize_data(self, input_dir: str) -> None:
        columns_config = get_config("input_adapters.format_normalizer.columns")
        os.makedirs(get_canonical_data_dir(), exist_ok=True)

        for file in os.listdir(input_dir):
            with open(os.path.join(get_canonical_data_dir(), file), "w") as output_file:
                writer = csv.writer(output_file)
                writer.writerow([c["name"] for c in columns_config])
                with open(os.path.join(input_dir, file), "r") as input_file:
                    for json_line in input_file.readlines():
                        tx = json.loads(json_line)
                        writer.writerow(
                            [
                                JsonsNormalizer.extract_column(tx, c["key"])
                                for c in columns_config
                            ]
                        )
