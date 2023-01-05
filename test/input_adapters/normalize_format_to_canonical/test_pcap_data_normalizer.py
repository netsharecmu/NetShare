import csv
import os
import shutil

from netshare.configs import set_config
from netshare.input_adapters.input_adapter_api import get_canonical_data_dir
from netshare.input_adapters.normalize_format_to_canonical.pcap_format_normalizer import (
    PcapNormalizer,
)


def test_normalize_data(tmp_path, test_data):
    normalizer = PcapNormalizer()

    # Create temporary directories
    source = tmp_path / "source"
    source.mkdir()
    work_folder = tmp_path / "work_folder"
    work_folder.mkdir()

    # Fill the source directory with files
    shutil.copy2(os.path.join(test_data, "http.cap"), str(source))
    set_config({"global_config": {"work_folder": str(work_folder)}})

    normalizer.normalize_data(str(source))

    with open(os.path.join(get_canonical_data_dir(), "http.csv"), "r") as csvfile:
        csv_data = csv.reader(csvfile)
        headers = next(csv_data)
        assert len(headers) == 15
        assert headers[:5] == ["srcip", "dstip", "srcport", "dstport", "proto"]
        first_packets = next(csv_data)
        assert len(first_packets) == 15
        assert first_packets[0] == "2449383661"
