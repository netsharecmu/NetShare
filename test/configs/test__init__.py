import pytest

from netshare.configs import get_config, set_config


@pytest.mark.parametrize(
    "config, path, default_value, expected",
    [
        ({}, "a", None, None),
        ({}, ["a"], None, None),
        ({}, ["a", "b"], None, None),
        ({"a": 1}, "a", None, 1),
        ({"a": 1}, ["a", "b"], None, 1),
        ({"b": 2}, ["a", "b"], None, 2),
        ({"a": 1, "b": 2}, ["a", "b"], None, 1),
        ({"a": {"b": 3}}, ["a.b"], None, 3),
        ({"a": {"b": 3}}, ["a.c"], None, None),
    ],
)
def test_get_config_happy_flow(config, path, default_value, expected):
    set_config(config)
    assert get_config(path, default_value=default_value) == expected


@pytest.mark.parametrize(
    "config, path",
    [
        ({}, ["a"]),
        ({}, ["a", "b"]),
        ({"c": 1}, ["a", "b"]),
    ],
)
def test_get_config_exception(config, path):
    set_config(config)
    with pytest.raises(ValueError):
        get_config(path)
