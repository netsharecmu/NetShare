import ctypes
import os

from netshare.pre_process.normalize_format_to_csv.base_format_normalizer import (
    DataFormatNormalizer,
)
import netshare.pre_post_processors.netshare
from netshare.utils import exec_cmd


class PcapNormalizer(DataFormatNormalizer):
    """
    This normalizer build CSV files out of pcap files.
    """

    def normalize_data(self, input_dir: str, target_dir: str) -> None:
        cwd = os.path.dirname(
            os.path.abspath(netshare.pre_post_processors.netshare.__file__)
        )
        cmd = f"cd {cwd} && ./sharedlib.sh"
        exec_cmd(cmd, wait=True)

        pcap2csv_func = ctypes.CDLL(os.path.join(cwd, "pcap2csv.so")).pcap2csv
        pcap2csv_func.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        for file_name in os.listdir(input_dir):
            csv_file = os.path.join(target_dir, f"{file_name.split('.')[0]}.csv")
            pcap2csv_func(
                os.path.join(input_dir, file_name).encode("utf-8"),
                csv_file.encode("utf-8"),
            )
