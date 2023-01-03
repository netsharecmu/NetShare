from addict import Dict

from netshare.configs import get_config

config = Dict(enabled=True)


def is_ray_enabled():
    result = get_config("global_config.ray_enabled", default_value=config.enabled)
    config.enabled = result  # For backward compatibility
    return result
