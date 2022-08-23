import netshare.ray as ray
import time


@ray.remote
def f1(x):
    print(f'f1: {x}')
    time.sleep(1)
    return x * x


@ray.remote(scheduling_strategy="SPREAD", max_calls=1)
def f2(x):
    print(f'f2: {x}')
    time.sleep(1)
    return x * 2


@ray.remote
def f3():
    print(f'f3')
    time.sleep(1)
    return 100


@ray.remote(scheduling_strategy="SPREAD", max_calls=1)
def f4():
    print(f'f4')
    time.sleep(1)
    return 101


if __name__ == '__main__':
    ray.config.enabled = False
    ray.init(address="auto")

    functions_to_test = [f1, f2]

    for function in functions_to_test:
        futures = [function.remote(i) for i in range(4)]
        print(ray.get(futures))

        futures = function.remote(10)
        print(ray.get(futures))

    for function in functions_to_test:
        futures = [function.remote(x=i) for i in range(4)]
        print(ray.get(futures))

        futures = function.remote(x=10)
        print(ray.get(futures))

    functions_to_test = [f3, f4]

    for function in functions_to_test:
        futures = [function.remote() for i in range(4)]
        print(ray.get(futures))

        futures = function.remote()
        print(ray.get(futures))

    ray.shutdown()
