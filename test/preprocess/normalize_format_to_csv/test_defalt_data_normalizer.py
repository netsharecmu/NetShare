import os

from netshare.preprocess.normalize_format_to_csv.default_format_normalizer import (
    CsvNormalizer,
)


def test_normalize_data(tmp_path):
    normalizer = CsvNormalizer()

    # Create temporary directories
    source = tmp_path / "source"
    source.mkdir()

    # Fill the source directory with files
    (source / "file1.txt").write_text("file1")
    (source / "file2.txt").write_text("file2")

    target = normalizer.normalize_data(str(source))

    # Check that the files were copied
    assert open(os.path.join(target, "file1.txt")).read() == "file1"
    assert open(os.path.join(target, "file2.txt")).read() == "file2"
