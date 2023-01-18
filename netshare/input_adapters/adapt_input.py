from netshare.input_adapters.data_source import fetch_data
from netshare.input_adapters.normalize_format_to_canonical import normalize_files_format


def adapt_input():
    """
    This is the main function of the input_adapters phase.
    We get the configuration, and prepare everything for the training phase.

    This function execute the following steps:
    1. Copy the data from the data source to local (e.g. from S3 bucket, DB, etc.)
    2. Normalize the files format to CSV (e.g. from pcap, json, etc.)
    """
    raw_data_dir = fetch_data()
    normalize_files_format(raw_data_dir)
