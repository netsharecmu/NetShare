import abc

from config_io import Config


class DataSource(abc.ABC):
    """
    Base class for all source adapters.
    A data source is the place where the data is stored, and we should read from.
    Example data sources are: local file system, S3, HTTP API, etc.
    """

    @abc.abstractmethod
    def fetch_data(self, target_dir: str) -> None:
        """
        This function reads the data from the source and stores it in the target_dir.
        The input config is a dynamic dictionary, and each data source will define its own.

        The data should be stored in the target_dir as files.
        """
        raise NotImplementedError()
