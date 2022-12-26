import pytest

from netshare.configs import get_config, set_config


@pytest.mark.parametrize(
    "config, path1, path2, default_value, expected",
    [
        ({}, "a", None, None, None),
        ({}, "a", "b", None, None),
        ({"a": 1}, "a", "b", None, 1),
        ({"b": 2}, "a", "b", None, 2),
        ({"a": 1, "b": 2}, "a", "b", None, 1),
        ({"a": {"b": 3}}, "a.b", None, None, 3),
    ],
)
def test_get_config_happy_flow(config, path1, path2, default_value, expected):
    set_config(config)
    assert get_config(path=path1, path2=path2, default_value=default_value) == expected


@pytest.mark.parametrize(
    "config, path1, path2",
    [
        ({}, "a", None),
        ({}, "a", "b"),
        ({"c": 1}, "a", "b"),
    ],
)
def test_get_config_exception(config, path1, path2):
    set_config(config)
    with pytest.raises(ValueError):
        get_config(path=path1, path2=path2)
