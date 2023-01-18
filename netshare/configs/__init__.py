import os
from typing import Any, List, Optional, Union

from config_io import Config

from netshare.utils.logger import logger

from .default import __path__ as defaults_path

_config: Optional[Config] = None


def load_from_file(path: str) -> None:
    global _config
    _config = Config.load_from_file(path, default_search_paths=defaults_path)


def set_config(config: Union[Config, dict]) -> None:
    global _config
    _config = config


def get_config(
    path: Optional[Union[str, List[str]]] = None,
    *,
    default_value: Any = Exception,
) -> Any:
    """
    Get config value by path.
    If the path is None, return the whole config.
    If the path is string, it will be split by ".", and we will get the value recursively.
    If the given path is a list of strings, we will consider each string as a possible path, and will
     return the first path that works. This option is mainly for backward compatibility of old configurations.
    If neither of the paths work, we will return the default value.
        If the default value is kept empty, we will raise an exception.
    """
    global _config
    if _config is None:
        raise ValueError("Config is not set")
    if path is None:
        return _config
    if isinstance(path, str):
        try:
            curr = _config
            for sep in path.split("."):
                curr = curr[sep]
            return curr
        except (KeyError, AttributeError):
            pass
    if isinstance(path, list):
        for p in path:
            try:
                return get_config(p)
            except ValueError:
                pass
    if default_value != Exception:
        return default_value
    raise ValueError(f"Config path {path} not found") from None


def change_work_folder(work_folder: Optional[str]) -> None:
    abs_path = os.path.expanduser(
        os.path.abspath(work_folder or get_config("global_config.work_folder"))
    )
    logger.info(f"Using work folder: {abs_path}")
    get_config("global_config")["work_folder"] = abs_path
