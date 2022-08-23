import ray
import sys

ray.init(address="auto")
# ray.init(address='ray://128.105.144.254:10001')

import time

@ray.remote
def f():
    time.sleep(0.01)
    ip = ray._private.services.get_node_ip_address()
    # f = open(f"/nfs/ray-test/node{ip}.txt", "a")
    # f.write("1")
    # f.close()
    return ip

# Get a list of the IP addresses of the nodes that have joined the cluster.
print("Python version")
print(sys.version)
print(set(ray.get([f.remote() for _ in range(100000)])))
print(ray.nodes())
