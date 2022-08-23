from .config import config as ray_config

def init(*args, **kwargs):
    if ray_config.enabled:
        print('Ray is enabled')
        import ray
        ray.init(*args, **kwargs)
    else:
        print('Ray is disabled')


def shutdown(*args, **kargs):
    if ray_config.enabled:
        print('Ray is enabled')
        import ray
        ray.shutdown(*args, **kargs)
    else:
        print("Ray is disabled")
