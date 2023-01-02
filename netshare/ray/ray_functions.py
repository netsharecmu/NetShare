from netshare.ray.ray_config import is_ray_enabled

_is_ray_initialized = False


def init(*args, **kwargs):
    global _is_ray_initialized
    if is_ray_enabled():
        import ray

        if not _is_ray_initialized:
            print("Ray is enabled")
            ray.init(*args, **kwargs)
            _is_ray_initialized = True
    else:
        if not _is_ray_initialized:
            print("Ray is enabled")
            _is_ray_initialized = True


def shutdown(*args, **kargs):
    global _is_ray_initialized
    if is_ray_enabled():
        import ray

        if _is_ray_initialized:
            ray.shutdown(*args, **kargs)
            _is_ray_initialized = False
