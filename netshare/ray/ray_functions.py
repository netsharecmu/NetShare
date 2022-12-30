from netshare.ray.ray_config import is_ray_enabled

_is_ray_initialized = False


def init(*args, **kwargs):
    global _is_ray_initialized
    if is_ray_enabled():
        print("Ray is enabled")
        import ray

        if not _is_ray_initialized:
            ray.init(*args, **kwargs)
            _is_ray_initialized = True
    else:
        print("Ray is disabled")


def shutdown(*args, **kargs):
    global _is_ray_initialized
    if is_ray_enabled():
        print("Ray is enabled")
        import ray

        if _is_ray_initialized:
            ray.shutdown(*args, **kargs)
            _is_ray_initialized = False
    else:
        print("Ray is disabled")
