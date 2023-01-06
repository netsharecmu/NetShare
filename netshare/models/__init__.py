from typing import Type

from netshare.configs import get_config

from .doppelganger_tf_model import DoppelGANgerTFModel  # type: ignore
from .model import Model


def build_model_from_config() -> Type[Model]:
    if get_config("model.class") == "DoppelGANgerTFModel":
        return DoppelGANgerTFModel  # type: ignore
    raise ValueError("Unknown model class")


__all__ = ["Model", "DoppelGANgerTFModel", "build_model_from_config"]
