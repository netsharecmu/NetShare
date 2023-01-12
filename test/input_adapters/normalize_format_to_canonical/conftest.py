import pytest

from netshare.configs import set_config


@pytest.fixture(autouse=True)
def work_filder(tmp_path):
    work_folder = tmp_path / "work_folder"
    work_folder.mkdir()
    set_config({"global_config": {"work_folder": str(work_folder)}})

    return str(work_folder)
