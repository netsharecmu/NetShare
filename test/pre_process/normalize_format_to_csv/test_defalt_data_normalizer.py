from netshare.pre_process.normalize_format_to_csv.default_format_normalizer import (
    CsvNormalizer,
)


def test_normalize_data(tmp_path):
    normalizer = CsvNormalizer()

    # Create temporary directories
    source = tmp_path / "source"
    source.mkdir()
    target = tmp_path / "target"
    target.mkdir()

    # Fill the source directory with files
    (source / "file1.txt").write_text("file1")
    (source / "file2.txt").write_text("file2")

    normalizer.normalize_data(str(source), str(target), {})

    # Check that the files were copied
    assert (target / "file1.txt").read_text() == "file1"
    assert (target / "file2.txt").read_text() == "file2"
