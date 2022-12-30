import os
from typing import Any, Optional, Union

from config_io import Config

from .default import __path__ as defaults_path

_config: Optional[Config] = None


def load_from_file(path: str) -> None:
    global _config
    _config = Config.load_from_file(path, default_search_paths=defaults_path)


def set_config(config: Union[Config, dict]) -> None:
    global _config
    _config = config


def get_config(
    path: Optional[str] = None,
    default_value: Any = Exception,
    path2: Optional[str] = None,
) -> Any:
    global _config
    if _config is None:
        raise ValueError("Config is not set")
    if path:
        try:
            curr = _config
            for sep in path.split("."):
                curr = curr[sep]
            return curr
        except (KeyError, AttributeError):
            if path2:
                return get_config(path=path2, default_value=default_value)
            if default_value != Exception:
                return default_value
            raise ValueError(f"Config path {path} not found") from None
    return _config


def change_work_folder(work_folder: Optional[str]) -> None:
    if work_folder:
        get_config()["global_config"]["work_folder"] = os.path.expanduser(work_folder)
