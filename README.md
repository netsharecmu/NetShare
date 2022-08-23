# Practical GAN-based Synthetic IP Header Trace Generation using NetShare

**Update 8/23/2022: We are currently porting full funtionalities from the branch [`camera-ready`](https://github.com/netsharecmu/NetShare/tree/camera-ready) as a python package. Please checkout [`camera-ready`](https://github.com/netsharecmu/NetShare/tree/camera-ready) branch if you would like to replicate all the experiments in the [paper](https://dl.acm.org/doi/abs/10.1145/3544216.3544251)**.

[[paper (SIGCOMM 2022)](https://dl.acm.org/doi/abs/10.1145/3544216.3544251)][[web service demo](https://drive.google.com/file/d/1vPuneEb14A2w7fKyCJ41NAHzsvpLQP5H/view)]

**Authors:** [[Yucheng Yin](https://sniperyyc.com/)] [[Zinan Lin](http://www.andrew.cmu.edu/user/zinanl/)] [[Minhao Jin](https://www.linkedin.com/in/minhao-jin-1328b8164/)] [[Giulia Fanti](https://www.andrew.cmu.edu/user/gfanti/)] [[Vyas Sekar](https://users.ece.cmu.edu/~vsekar/)]

**Abstract:** We explore the feasibility of using Generative Adversarial Networks (GANs) to automatically learn generative models to generate synthetic packet- and flow header traces for network-ing tasks (e.g., telemetry, anomaly detection, provisioning). We identify key fidelity, scalability, and privacy challenges and tradeoffs in existing GAN-based approaches. By synthesizing domain-specific insights with recent advances in machine learning and privacy, we identify design choices to tackle these challenges. Building on these insights, we develop an end-to-end framework, NetShare. We evaluate NetShare on six diverse packet header traces and find that: (1) across distributional metrics and traces, it achieves 46% more accuracy than baselines, and (2) it meets users’ requirements of downstream tasks in evaluating accuracy and rank ordering of candidate approaches.

# Installation/Setup
## Step 1: Install NetShare Python package (Required)
We recommend installing NetShare in a virtual environment (e.g., Anaconda3). We test with virtual environment with Python==3.6.

```
git clone https://github.com/netsharecmu/NetShare.git
cd NetShare/
pip3 install -e .
```

## Step 2: How to start Ray? (Optional but **strongly** recommended)
Ray is a unified framework for scaling AI and Python applications. Our framework utilizes Ray to increase parallelism and distribute workloads among the cluster automatically and efficiently.

## Laptop/Single-machine (only recommended for demo/dev/fun)
```
ray start --head --port=6379 --include-dashboard=True --dashboard-host=0.0.0.0 --dashboard-port=8265
```

Please go to [http://localhost:8265](http://localhost:8265) to view the Ray dashboard.

<!-- <p align="center">
  <img width="1000" src="doc/figs/ray_dashboard_example.png">
</p>
<p align="center">
  Figure 1: Example of Ray Dashboard
</p> -->

## Multi-machines (**strongly** recommended for faster training/generation)
We provide a utility script and [README](util/README.md) under `util/` for setting up a Ray cluster. As a reference, we are using [Cloudlab](https://www.cloudlab.us/) which is referred as ``custom cluster'' in the Ray documentation. If you are using a different cluster (e.g., AWS, GCP, Azure), please refer to the [Ray doc](https://docs.ray.io/en/releases-2.0.0rc0/cluster/cloud.html#cluster-cloud) for full reference.

# Datasets
***We are adding more datasets! Feel free to add your own and contribute!***

Our paper uses **six** public datasets for reproducibility. Please to refer to the [README](traces/1M/README.md) for detailed descriptions of the datasets.

Please download the six datasets as tar.gz file [here](https://drive.google.com/file/d/19neMv1iXjnMn5IdpPoupdyf4Yy4GvL7R/view?usp=sharing) and unzip to `traces`.

# Example usage
***We are adding more examples of usage (PCAP, NetFlow, w/ and w/o DP). Please stay tuned!***

Here is a minimal working example to generate synthetic PCAP files without differential privacy. Please refer to [`examples`](examples/) for scripts and config files.

[Driver code](examples/pcap/pcap_nodp.py)
```Python
import netshare.ray as ray
from netshare import Generator
from config_io import Config

if __name__ == '__main__':
    # Change to False if you would not like to use Ray
    ray.config.enabled = True
    ray.init(address="auto")

    # configuration file
    generator = Generator(config="config_example_pcap_nodp.json")

    # where you would like to store your intermediate files and results
    generator.train_and_generate(work_folder='results/test')

    ray.shutdown()
```

The corresponding [configuration file](examples/pcap/config_example_pcap_nodp.json):
```json
{
    "global_config": {
        "original_data_file": "../../traces/1M/caida/raw.pcap",
        "dataset_type": "pcap",
        "n_chunks": 10,
        "dp": false
    },
    "default": "pcap.json"
}
```

Notice that we provide a bunch of [default configurations](netshare/configs/default) for different datasets/training mechanisms. In most cases you only need to write a few lines of configs.


# References
Please cite our paper approriately if you find NetShare is useful!

```bibtex
@inproceedings{netshare-sigcomm2022,
  author = {Yin, Yucheng and Lin, Zinan and Jin, Minhao and Fanti, Giulia and Sekar, Vyas},
  title = {Practical GAN-Based Synthetic IP Header Trace Generation Using NetShare},
  year = {2022},
  isbn = {9781450394208},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3544216.3544251},
  doi = {10.1145/3544216.3544251},
  abstract = {We explore the feasibility of using Generative Adversarial Networks (GANs) to automatically learn generative models to generate synthetic packet- and flow header traces for networking tasks (e.g., telemetry, anomaly detection, provisioning). We identify key fidelity, scalability, and privacy challenges and tradeoffs in existing GAN-based approaches. By synthesizing domain-specific insights with recent advances in machine learning and privacy, we identify design choices to tackle these challenges. Building on these insights, we develop an end-to-end framework, NetShare. We evaluate NetShare on six diverse packet header traces and find that: (1) across all distributional metrics and traces, it achieves 46% more accuracy than baselines and (2) it meets users' requirements of downstream tasks in evaluating accuracy and rank ordering of candidate approaches.},
  booktitle = {Proceedings of the ACM SIGCOMM 2022 Conference},
  pages = {458–472},
  numpages = {15},
  keywords = {privacy, synthetic data generation, network packets, network flows, generative adversarial networks},
  location = {Amsterdam, Netherlands},
  series = {SIGCOMM '22}
}
```

Part of the source code is adapated from the following open-source projects:

- [DoppelGANger](https://github.com/fjxmlzn/DoppelGANger)
- [GPUTaskScheduler](https://github.com/fjxmlzn/GPUTaskScheduler)
- [BSN](https://github.com/fjxmlzn/BSN)
- [Ray](https://github.com/ray-project/ray)
- [config_io](https://github.com/fjxmlzn/config_io)
