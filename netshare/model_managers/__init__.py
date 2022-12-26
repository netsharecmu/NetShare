from config_io import Config

from netshare.configs import get_config

from .model_manager import ModelManager
from .netshare_manager.netshare_manager import NetShareManager
from .dg_model_manager import DGModelManager


def build_model_manager_from_config() -> ModelManager:
    if get_config("model_manager.class") == "NetShareManager":
        model_manager_class = NetShareManager
    elif get_config("model_manager.class") == "DGModelManager":
        model_manager_class = DGModelManager
    else:
        raise ValueError("Unknown model manager class")
    model_manager_config = Config(get_config("global_config"))
    model_manager_config.update(get_config("model_manager.config"))
    return model_manager_class(config=model_manager_config)  # type: ignore


__all__ = [
    "ModelManager",
    "NetShareManager",
    "DGModelManager",
    "build_model_manager_from_config",
]
