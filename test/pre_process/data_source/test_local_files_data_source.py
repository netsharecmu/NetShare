from netshare.pre_process.data_source.local_files_data_source import (
    LocalFilesDataSource,
)


def test_fetch_data(tmp_path):
    data_source = LocalFilesDataSource()

    # Create temporary directories
    source = tmp_path / "source"
    source.mkdir()
    target = tmp_path / "target"
    target.mkdir()

    # Fill the source directory with files
    (source / "file1.txt").write_text("file1")
    (source / "file2.txt").write_text("file2")
    (source / "subdir").mkdir()
    (source / "subdir" / "file3.txt").write_text("file3")

    data_source.fetch_data({"input_folder": str(source)}, str(target))

    # Check that the files were copied
    assert (target / "file1.txt").read_text() == "file1"
    assert (target / "file2.txt").read_text() == "file2"
    assert (target / "subdir_file3.txt").read_text() == "file3"
