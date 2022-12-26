from netshare.configs import get_config
from netshare.logger import logger

from .base_data_source import DataSource
from .local_files_data_source import LocalFilesDataSource
from .s3_data_source import S3DataSource


def fetch_data() -> str:
    """
    This is the builder function for the data source.
    It returns the path to the directory that contains the fetched data.
    """
    data_source_config = get_config("pre_process.data_source", default_value={})
    data_source_type = data_source_config.get("type") or "local_files"

    data_source: DataSource
    if data_source_type == "s3":
        logger.info("Fetching data from S3")
        data_source = S3DataSource()
    elif data_source_type == "local_files":
        logger.info("Fetching data from local files")
        data_source = LocalFilesDataSource()
    else:
        raise ValueError(f"Unknown data source type: {data_source_type}")

    return data_source.fetch_data()
