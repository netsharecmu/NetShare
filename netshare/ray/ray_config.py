from addict import Dict

from netshare.configs import get_config

config = Dict(enabled=True)
config.freeze()


def is_ray_enabled():
    return get_config("global_config.ray_enabled", default_value=config.enabled)
