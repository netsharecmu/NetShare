import csv
import os
import shutil

from netshare.preprocess.normalize_format_to_csv.pcap_format_normalizer import (
    PcapNormalizer,
)


def test_normalize_data(tmp_path, test_data):
    normalizer = PcapNormalizer()

    # Create temporary directories
    source = tmp_path / "source"
    source.mkdir()

    # Fill the source directory with files
    shutil.copy2(os.path.join(test_data, "http.cap"), str(source))

    target = normalizer.normalize_data(str(source))

    with open(os.path.join(target, "http.csv"), "r") as csvfile:
        csv_data = csv.reader(csvfile)
        headers = next(csv_data)
        assert len(headers) == 15
        assert headers[:5] == ["srcip", "dstip", "srcport", "dstport", "proto"]
        first_packets = next(csv_data)
        assert len(first_packets) == 15
        assert first_packets[0] == "2449383661"
