import functools

from netshare.configs import get_config, set_config
from netshare.utils.ray.ray_config import is_ray_enabled


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
        raise TypeError("Remote functions cannot be called directly.")

    def remote(self, *args, **kwargs):
        if is_ray_enabled():
            if self._actual_remote_function is None:
                original_func, *rest_args = self._ray_args

                def load_config_and_call(*inner_args, **inner_kwargs):
                    """
                    In this function, we make sure that the new ray process will have
                        the same config as the original process.
                    """
                    set_config(inner_kwargs.pop("local_config"))
                    return original_func(*inner_args, **inner_kwargs)

                self._ray_args = (load_config_and_call, *rest_args)
                import ray

                if len(self._ray_kwargs) == 0:
                    self._actual_remote_function = ray.remote(
                        *self._ray_args, **self._ray_kwargs
                    )
                else:
                    self._actual_remote_function = ray.remote(**self._ray_kwargs)(
                        *self._ray_args
                    )
            kwargs["local_config"] = get_config()
            return self._actual_remote_function.remote(*args, **kwargs)
        else:
            return ResultWrapper(self._ray_args[0](*args, **kwargs))


def remote(*args, **kwargs):
    if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
        # This is the case where the decorator is just @ray.remote.
        # "args[0]" is the class or function under the decorator.
        return RemoteFunctionWrapper(args[0])
    if not (len(args) == 0 and len(kwargs) > 0):
        raise ValueError("Error in the parameters of the decorator")
    return functools.partial(RemoteFunctionWrapper, **kwargs)


def get(object_refs, **kwargs):
    if is_ray_enabled():
        import ray

        return ray.get(object_refs, **kwargs)
    else:
        if isinstance(object_refs, ResultWrapper):
            return object_refs.get_result()
        elif isinstance(object_refs, list):
            return [object_ref.get_result() for object_ref in object_refs]
