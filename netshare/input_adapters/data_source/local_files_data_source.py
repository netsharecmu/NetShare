import os
import shutil
import tempfile

from netshare.configs import get_config
from netshare.input_adapters.data_source import DataSource


class LocalFilesDataSource(DataSource):
    """
    This is the simple source adapter that reads the data from the local file system and put it as is in the target_dir.
    Subdirectories are being flattened, and the filenames are change to be unique.

    The config should contain the following:
    * input_folder: The path to the folder that contains the data.
    """

    def fetch_data(self) -> str:
        target_dir = tempfile.mkdtemp()
        single_file = get_config(
            [
                "global_config.original_data_file",
                "input_adapters.data_source.original_data_file",
            ],
            default_value=None,
        )
        if single_file:
            if not os.path.exists(single_file):
                raise ValueError(f"Input file {single_file} does not exist")
            shutil.copy(single_file, target_dir)
            return target_dir

        input_folder = get_config(
            [
                "global_config.original_data_folder",
                "input_adapters.data_source.input_folder",
            ],
        )
        if not input_folder or not isinstance(input_folder, str):
            raise ValueError(
                "Missing input location in the config (either original_data_folder or original_data_file)"
            )
        if not os.path.exists(input_folder):
            raise ValueError(f"Input directory {input_folder} does not exist")
        for root, dirs, files in os.walk(input_folder):
            for filename in files:
                unique_filename = (
                    os.path.join(root, filename)
                    .replace(input_folder, "")
                    .replace("/", "__")
                    .lstrip("_")
                )
                shutil.copyfile(
                    os.path.join(root, filename),
                    os.path.join(target_dir, unique_filename),
                )
        return target_dir
