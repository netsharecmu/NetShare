import abc


class DataFormatNormalizer(abc.ABC):
    """
    Base class for all data normalizer.
    A data normalizer is the logic that processes the input files with various formats, and generate a file
        from our canonical format. Our current canonical format is CSV.
    Example file formats are: pcap, json, etc.
    """

    @abc.abstractmethod
    def normalize_data(self, input_dir: str) -> str:
        """
        This function normalize the data from the input_dir and stores it in the target_dir.
        For every file in the given directory it writes a new file with the same name in the target directory,
            that contains the same data in a CSV format.

        The input config is a dynamic dictionary, and each format normalizer will define its own.
        :return: The path to the directory that contains the normalized data.
        """
        raise NotImplementedError()
