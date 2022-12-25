from netshare.logger import logger
from netshare.configs import get_config

from .local_files_data_source import LocalFilesDataSource
from .s3_data_source import S3DataSource
from .base_data_source import DataSource


def fetch_data(target_dir: str) -> None:
    """
    This is the builder function for the data source.
    """
    data_source_config = get_config().get("pre_process", {}).get("data_source", {})
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

    data_source.fetch_data(target_dir)
