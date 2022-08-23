import functools
from .config import config as ray_config


class ResultWrapper(object):
    def __init__(self, result):
        self._result = result

    def get_result(self):
        return self._result


class RemoteFunctionWrapper(object):
    def __init__(self, *args, **kwargs):
        self._actual_remote_function = None
        self._ray_args = args
        self._ray_kwargs = kwargs

    def __call__(self, *args, **kwargs):
        raise TypeError('Remote functions cannot be called directly.')

    def remote(self, *args, **kwargs):
        if ray_config.enabled:
            if self._actual_remote_function is None:
                import ray
                if len(self._ray_kwargs) == 0:
                    self._actual_remote_function = ray.remote(
                        *self._ray_args, **self._ray_kwargs)
                else:
                    self._actual_remote_function = ray.remote(
                        **self._ray_kwargs)(*self._ray_args)
            return self._actual_remote_function.remote(*args, **kwargs)
        else:
            return ResultWrapper(self._ray_args[0](*args, **kwargs))


def remote(*args, **kwargs):
    if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
        # This is the case where the decorator is just @ray.remote.
        # "args[0]" is the class or function under the decorator.
        return RemoteFunctionWrapper(args[0])
    if not (len(args) == 0 and len(kwargs) > 0):
        raise ValueError('Error in the parameters of the decorator')
    return functools.partial(RemoteFunctionWrapper, **kwargs)


def get(object_refs, **kwargs):
    if ray_config.enabled:
        import ray
        return ray.get(object_refs, **kwargs)
    else:
        if isinstance(object_refs, ResultWrapper):
            return object_refs.get_result()
        elif isinstance(object_refs, list):
            return [object_ref.get_result() for object_ref in object_refs]
