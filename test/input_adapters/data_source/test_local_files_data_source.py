import os

from netshare.configs import set_config
from netshare.input_adapters.data_source.local_files_data_source import (
    LocalFilesDataSource,
)


def test_fetch_data(tmp_path):
    data_source = LocalFilesDataSource()

    # Create temporary directories
    source = tmp_path / "source"
    source.mkdir()

    # Fill the source directory with files
    (source / "file1.txt").write_text("file1")
    (source / "file2.txt").write_text("file2")
    (source / "subdir").mkdir()
    (source / "subdir" / "file3.txt").write_text("file3")

    set_config({"global_config": {"original_data_folder": str(source)}})
    target = data_source.fetch_data()

    # Check that the files were copied
    assert open(os.path.join(target, "file1.txt"), "r").read() == "file1"
    assert open(os.path.join(target, "file2.txt"), "r").read() == "file2"
    assert open(os.path.join(target, "subdir_file3.txt"), "r").read() == "file3"
