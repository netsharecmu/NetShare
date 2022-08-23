# Practical GAN-based Synthetic IP Header Trace Generation using NetShare

**Update 8/23/2022: We are currently porting full funtionalities from the branch `camera-ready` as a python package. Please checkout `camera-ready` branch if you would like to replicate all the experiments in the [paper](https://dl.acm.org/doi/abs/10.1145/3544216.3544251)**.

[[paper (SIGCOMM 2022)](https://dl.acm.org/doi/abs/10.1145/3544216.3544251)][[web service demo](https://drive.google.com/file/d/1vPuneEb14A2w7fKyCJ41NAHzsvpLQP5H/view)]

**Authors:** [[Yucheng Yin](https://sniperyyc.com/)] [[Zinan Lin](http://www.andrew.cmu.edu/user/zinanl/)] [[Minhao Jin](https://www.linkedin.com/in/minhao-jin-1328b8164/)] [[Giulia Fanti](https://www.andrew.cmu.edu/user/gfanti/)] [[Vyas Sekar](https://users.ece.cmu.edu/~vsekar/)]

**Abstract:** We explore the feasibility of using Generative Adversarial Networks (GANs) to automatically learn generative models to generate synthetic packet- and flow header traces for network-ing tasks (e.g., telemetry, anomaly detection, provisioning). We identify key fidelity, scalability, and privacy challenges and tradeoffs in existing GAN-based approaches. By synthesizing domain-specific insights with recent advances in machine learning and privacy, we identify design choices to tackle these challenges. Building on these insights, we develop an end-to-end framework, NetShare. We evaluate NetShare on six diverse packet header traces and find that: (1) across distributional metrics and traces, it achieves 46% more accuracy than baselines, and (2) it meets users’ requirements of downstream tasks in evaluating accuracy and rank ordering of candidate approaches.

# Installation/Setup
## Step 1: Install NetShare Python package (Required)
We recommend installing NetShare in a virtual environment (e.g., Anaconda3). We test with virtual environment with Python==3.6.

```
pip3 install NetShare
```

## Step 2: How to start Ray? (Optional but **strongly** recommended)
Ray is a unified framework for scaling AI and Python applications. Our framework utilizes Ray to increase parallelism and distribute workloads among the cluster automatically and efficiently.

## Laptop/Single-machine (only recommended for demo/dev/fun)
```
ray start --head --port=6379 --include-dashboard=True --dashboard-host=0.0.0.0 --dashboard-port=8265
```

Please go to [http://localhost:8265](http://localhost:8265) to view the Ray dashboard.

<p align="center">
  <img width="1000" src="doc/figs/ray_dashboard_example.png">
</p>
<p align="center">
  Figure 1: Example of Ray Dashboard
</p>

## Multi-machines (**strongly** recommended for faster training/generation)
We provide a utility script and [README](util/README.md) under `util/` for setting up a Ray cluster. As a reference, we are using [Cloudlab](https://www.cloudlab.us/) which is referred as ``custom cluster'' in the Ray documentation. If you are using a different cluster (e.g., AWS, GCP, Azure), please refer to the [Ray doc](https://docs.ray.io/en/releases-2.0.0rc0/cluster/cloud.html#cluster-cloud) for full reference.

# Tests

## Ray

```
python -m tests.ray.test_ray_disabled
```

```
python -m tests.ray.test_ray_enabled
```

## Framework

```
python -m tests.framework.test_ray_disabled
```

```
python -m tests.framework.test_ray_enabled
```
