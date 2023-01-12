import csv
import os

import pytest

from netshare.configs import get_config
from netshare.input_adapters.input_adapter_api import get_canonical_data_dir
from netshare.input_adapters.normalize_format_to_canonical.jsons_normalizer import (
    JsonsNormalizer,
)


def test_normalize_data(tmp_path, test_data):
    normalizer = JsonsNormalizer()
    source = tmp_path / "source"
    source.mkdir()

    (source / "file1.txt").write_text('{"a": 1, "b": 2}\n{"a": 3, "b": 4}\n')
    (source / "file2.txt").write_text('{"a": 5, "b": 6}\n{"a": 7, "b": 8}\n')
    get_config()["input_adapters"] = {
        "format_normalizer": {
            "columns": [
                {"name": "first", "key": "a"},
                {"name": "second", "key": "b"},
            ]
        }
    }

    normalizer.normalize_data(str(source))

    with open(os.path.join(get_canonical_data_dir(), "file1.txt"), "r") as csvfile:
        csv_data = csv.reader(csvfile)
        headers = next(csv_data)
        assert headers == ["first", "second"]
        assert next(csv_data) == ["1", "2"]
        assert next(csv_data) == ["3", "4"]

    with open(os.path.join(get_canonical_data_dir(), "file2.txt"), "r") as csvfile:
        csv_data = csv.reader(csvfile)
        headers = next(csv_data)
        assert headers == ["first", "second"]
        assert next(csv_data) == ["5", "6"]
        assert next(csv_data) == ["7", "8"]


@pytest.mark.parametrize(
    "input_json, key, expected",
    [
        ({"a": 1, "b": 2}, "a", 1),
        ({"a": 1, "b": 2}, "b", 2),
        ({"a": {"c": 3}, "b": 2}, "a.c", 3),
        ({"a": {"c": 3}, "b": 2}, "b", 2),
    ],
)
def test_extract_column(input_json, key, expected):
    assert JsonsNormalizer.extract_column(input_json, key) == expected


def test_extract_column_invalid_key():
    with pytest.raises(KeyError):
        JsonsNormalizer.extract_column({}, "a")


def test_extract_column_invalid_result():
    with pytest.raises(ValueError):
        JsonsNormalizer.extract_column({"a": {"b": 1}}, "a")
