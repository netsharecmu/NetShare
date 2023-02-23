import ctypes
import os

import netshare.input_adapters.normalize_format_to_canonical.pcap_c_files
from netshare.input_adapters.input_adapter_api import get_canonical_data_dir
from netshare.input_adapters.normalize_format_to_canonical.base_format_normalizer import (
    DataFormatNormalizer,
)
from netshare.utils import exec_cmd


class PcapNormalizer(DataFormatNormalizer):
    """
    This normalizer build CSV files out of pcap files.
    """

    def normalize_data(self, input_dir: str) -> None:
        cwd = os.path.dirname(
            os.path.abspath(
                netshare.input_adapters.normalize_format_to_canonical.pcap_c_files.__file__
            )
        )
        cmd = f"cd {cwd} && ./sharedlib.sh"
        exec_cmd(cmd, wait=True)

        pcap2csv_func = ctypes.CDLL(os.path.join(cwd, "pcap2csv.so")).pcap2csv
        pcap2csv_func.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        os.makedirs(get_canonical_data_dir(), exist_ok=True)
        for file_name in os.listdir(input_dir):
            csv_file = os.path.join(
                get_canonical_data_dir(), f"{file_name.split('.')[0]}.csv"
            )
            pcap2csv_func(
                os.path.join(input_dir, file_name).encode("utf-8"),
                csv_file.encode("utf-8"),
            )
