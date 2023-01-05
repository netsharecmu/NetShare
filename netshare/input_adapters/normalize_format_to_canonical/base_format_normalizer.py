import abc


class DataFormatNormalizer(abc.ABC):
    """
    Base class for all data normalizer.
    A data normalizer is the logic that processes the input files with various formats, and generate a file
        from our canonical format. Our current canonical format is CSV.
    Example file formats are: pcap, json, etc.
    """

    @abc.abstractmethod
    def normalize_data(self, input_dir: str) -> None:
        """
        This function normalize the data from the input_dir and stores it using the input_adapter_api.

        The configuration that is using by this adapter is dynamic, and each format normalizer will define its own.
        """
        raise NotImplementedError()
