import os.path

import pytest


@pytest.fixture
def test_data() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "test_data"))
