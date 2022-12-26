from typing import Type

from netshare.configs import get_config

from .model import Model
from .doppelganger_tf_model import DoppelGANgerTFModel


def build_model_from_config() -> Type[Model]:
    if get_config("model.class") == "DoppelGANgerTFModel":
        return DoppelGANgerTFModel
    raise ValueError("Unknown model class")


__all__ = ["Model", "DoppelGANgerTFModel", "build_model_from_config"]
