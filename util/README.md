# Cluster prerequisite
1. Nodes mounted with nfs.
2. Nodes can communicate with each other using ssh in normal user mode.

# Ray cluster setup

Launch ray cluster (Run this command on your own laptop or on the host in cluster)

When launching ray cluster on your own laptop or in the cluster, please make sure to activate the conda environment "NetShare".


```bash
# Change the host and workers ip in example.yaml
(NetShare) node1:/nfs/NetShare-dev$ export LC_ALL=C.UTF-8
(NetShare) node1:/nfs/NetShare-dev$ ray up ray/example.yaml
```

If launching the cluster from the cluster returns error like the following `FileNotFoundError: [Errno 2] No such file or directory: '/tmp/ray/cluster-test.lock'` it may be a bug.

Solution is:

```bash
# Change the host and workers ip in example.yaml
(NetShare) node1:/nfs/NetShare-dev$ export LC_ALL=C.UTF-8
(NetShare) node1:/nfs/NetShare-dev$ ray start --head
(NetShare) node1:/nfs/NetShare-dev$ ray stop
(NetShare) node1:/nfs/NetShare-dev$ ray up ray/example.yaml
```

Check if the ray cluster has been launched successfully.
``` bash
(NetShare) node1:/nfs/NetShare-dev$ ray status

======== Autoscaler status: 2022-07-23 10:08:03.979944 ========
Node status
---------------------------------------------------------------
Healthy:
 4 local.cluster.node
Pending:
 (no pending nodes)
Recent failures:
 (no failures)

Resources
---------------------------------------------------------------
Usage:
 0.0/160.0 CPU
 0.00/513.323 GiB memory
 0.14/223.987 GiB object_store_memory

Demands:
 (no resource demands)
```

Or 

``` bash
(NetShare) node1:/nfs/NetShare-dev$ python3 ray/check_nodes.py

Python version
3.6.13 |Anaconda, Inc.| (default, Jun  4 2021, 14:25:59)
[GCC 7.5.0]
{'128.105.144.191', '128.105.144.190', '128.105.144.179', '128.105.144.199'}
[{'NodeID': '58433beab7f1653cde1324b9a6764596fb0ef534eaf2182946ef28a4', 'Alive': True, 'NodeManagerAddress': '128.105.144.199', 'NodeManagerHostname': 'node4.env-test.cloudmigration-pg0.wisc.cloudlab.us', 'NodeManagerPort': 42365, 'ObjectManagerPort': 34065, 'ObjectStoreSocketName': '/tmp/ray/session_2022-07-22_12-34-30_640417_1124/sockets/plasma_store', 'RayletSocketName': '/tmp/ray/session_2022-07-22_12-34-30_640417_1124/sockets/raylet', 'MetricsExportPort': 50427, 'NodeName': '128.105.144.199', 'alive': True, 'Resources': {'object_store_memory': 60156551577.0, 'node:128.105.144.199': 1.0, 'memory': 140365287015.0, 'CPU': 40.0}}, {'NodeID': '315f7a09c9e7633d7e6119730004188116696c069a463472671018c5', 'Alive': True, 'NodeManagerAddress': '128.105.144.191', 'NodeManagerHostname': 'node3.env-test.cloudmigration-pg0.wisc.cloudlab.us', 'NodeManagerPort': 36329, 'ObjectManagerPort': 41259, 'ObjectStoreSocketName': '/tmp/ray/session_2022-07-22_12-34-30_640417_1124/sockets/plasma_store', 'RayletSocketName': '/tmp/ray/session_2022-07-22_12-34-30_640417_1124/sockets/raylet', 'MetricsExportPort': 58422, 'NodeName': '128.105.144.191', 'alive': True, 'Resources': {'CPU': 40.0, 'memory': 140363113677.0, 'object_store_memory': 60155620147.0, 'node:128.105.144.191': 1.0}}, {'NodeID': '30a870e576b48152b1150ca7d026ad9d51a16377121ad494355e7f76', 'Alive': True, 'NodeManagerAddress': '128.105.144.190', 'NodeManagerHostname': 'node2.env-test.cloudmigration-pg0.wisc.cloudlab.us', 'NodeManagerPort': 35237, 'ObjectManagerPort': 33677, 'ObjectStoreSocketName': '/tmp/ray/session_2022-07-22_12-34-30_640417_1124/sockets/plasma_store', 'RayletSocketName': '/tmp/ray/session_2022-07-22_12-34-30_640417_1124/sockets/raylet', 'MetricsExportPort': 56875, 'NodeName': '128.105.144.190', 'alive': True, 'Resources': {'object_store_memory': 60154269696.0, 'node:128.105.144.190': 1.0, 'memory': 140359962624.0, 'CPU': 40.0}}, {'NodeID': '3a36f6e72af22d38d74f353ef6daf44a02f25668875b528c462d2f17', 'Alive': True, 'NodeManagerAddress': '128.105.144.179', 'NodeManagerHostname': 'node1.env-test.cloudmigration-pg0.wisc.cloudlab.us', 'NodeManagerPort': 44069, 'ObjectManagerPort': 44719, 'ObjectStoreSocketName': '/tmp/ray/session_2022-07-22_12-34-30_640417_1124/sockets/plasma_store', 'RayletSocketName': '/tmp/ray/session_2022-07-22_12-34-30_640417_1124/sockets/raylet', 'MetricsExportPort': 65331, 'NodeName': '128.105.144.179', 'alive': True, 'Resources': {'object_store_memory': 60037563187.0, 'node:128.105.144.179': 1.0, 'CPU': 40.0, 'memory': 130087647437.0}}]
```

Check if dashboard has been launched successfully

dashboard: http://<host_public_ip>:8265/