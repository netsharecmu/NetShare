import os.path

import pytest

from netshare.configs import set_config


@pytest.fixture
def test_data() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "test_data"))


@pytest.fixture
def work_folder(tmp_path):
    work_folder = tmp_path / "work_folder"
    work_folder.mkdir()
    set_config({"global_config": {"work_folder": str(work_folder)}})
    return str(work_folder)
