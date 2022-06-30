# Practical GAN-based Synthetic IP Header Trace Generation using NetShare

**This branch has been DEPRECATED and ONLY served as backup for camera-ready experiments. Please checkout `main` branch for latest integration/codebase. Thanks!**

**Authors:** [[Yucheng Yin](https://sniperyyc.com/)] [[Zinan Lin](http://www.andrew.cmu.edu/user/zinanl/)] [[Minhao Jin](https://www.linkedin.com/in/minhao-jin-1328b8164/)] [[Giulia Fanti](https://www.andrew.cmu.edu/user/gfanti/)] [[Vyas Sekar](https://users.ece.cmu.edu/~vsekar/)]

**Abstract:** We explore the feasibility of using Generative Adversarial Networks (GANs) to automatically learn generative models to generate synthetic packet- and flow header traces for network-ing tasks (e.g., telemetry, anomaly detection, provisioning). We identify key fidelity, scalability, and privacy challenges and tradeoffs in existing GAN-based approaches. By synthesizing domain-specific insights with recent advances in machine learning and privacy, we identify design choices to tackle these challenges. Building on these insights, we develop an end-to-end framework, NetShare. We evaluate NetShare on six diverse packet header traces and find that: (1) across distributional metrics and traces, it achieves 46% more accuracy than baselines, and (2) it meets users’ requirements of downstream tasks in evaluating accuracy and rank ordering of candidate approaches.


# Setup
## Single-machine setup
Single-machine is only recommended for very small datasets and quick validation/prototype as GANs are very computationally expensive. We recommend using virtual environment to avoid conflicts (e.g., Anaconda).

```Bash
# Assume Anaconda is installed
# create virtual environment
conda create --name NetShare python=3.6

# installing dependencies
cd util/
pip3 install -r requirements.txt
```

## Multi-machine setup
Multi-machine setup is recommended for using NetShare. It is a bit harder to be set up compared with single-machine version but provides much more scalability and speed-up.

Our setup has been tested on Ubuntu 18.04 with Python 3.6 and Tensorflow 1.15 (CPU version). We utilize multiple powerful CPU servers with Network File System (NFS) setup on [Cloudlab](https://www.cloudlab.us) to achieve more parallelism. We utilize a controller-worker fashion where the controller is responsible for issuing commands to each worker while each worker does the actual job (e.g., training, generation).

Any similar cluster should work as long as it meets the following prerequisites:

- Multiple (powerful) CPU servers. As a reference, our testbed includes 10 [Cloudlab](https://www.cloudlab.us) machines (plus one controller machine) where each machine has Two Intel Xeon Silver 4114 10-core CPUs at 2.20 GHz and 192GB DDR4 memory. NetShare or GANs are in general very computationally-expensive.
- Setup NFS among servers so that they share the same storage space (NFS is natively supported on Cloudlab but you may refer to any NFS setup guide if you are using your own set of clusters.)
- Different servers can ssh into each other with **root privilege**. [Cloudlab](https://www.cloudlab.us) natively supports this feature.

To set up multiple machines, we provide scripts which you can launch locally from your laptop:

- Installing dependencies on each server
    ```Bash
    # Please modify the corresponding username/hostname etc. in util/setup_node_parallel.sh
    cd util
    bash setup_node_parallel.sh
    ```

- Get each server's (public) IP (so that the controller can ssh into different workers)
    ```Bash
    # Please modify the correspoding username/hostname etc. in util/get_ip.sh
    cd util
    bash get_ip.sh
    cp measurers.ini ../src/measurer_ini
    ```


# Dataset preparation
## Description
We use six public datasets for reproducibility. To be more specific,

Three NetFlow datasets:

1. [UGR16](https://nesg.ugr.es/nesg-ugr16/) dataset consists of traffic (including attacks) from NetFlow v9 collectors in a Spanish ISP network. We used data from the third week of March 2016. 
2. [CIDDS](https://www.hs-coburg.de/forschung/forschungsprojekte-oeffentlich/informationstechnologie/cidds-coburg-intrusion-detection-data-sets.html) dataset emulates a small business environment with several clients and servers (e.g., email, web) with injected malicious traffic was executed. Each NetFlow entry recorded with the label (benign/attack) and attack type (DoS, brute force, port scan). 
3. [TON](https://research.unsw.edu.au/projects/toniot-datasets) dataset represents telemetry IoT sensors. We use a sub-dataset (“Train_Test_datasets”) for evaluating cybersecurity-related ML algorithms; of its 461,013 records, 300,000 (65.07%) are normal, and the rest (34.93%) combine nine evenly-distributed attack types (e.g., backdoor, DDoS, injection, MITM).

Three PCAP datasets:

1. [CAIDA](https://www.caida.org/catalog/datasets/passive_dataset/) contains anonymized traces from high-speed monitors on a commercial backbone link. Our subset is from the New York collector in March 2018. (**Require an CAIDA account to download the data**)
2. [DC](https://pages.cs.wisc.edu/~tbenson/IMC10_Data.html) dataset is a packet capture from the "UNI1" data center studied in the [IMC 2010 paper](https://pages.cs.wisc.edu/~tbenson/papers/imc192.pdf).
3. [CA](https://www.netresec.com/?page=MACCDC) dataset is traces from The U.S. National CyberWatch Mid-Atlantic Collegiate Cyber Defense Competitions from March 2012.

## Download [5 mins]

Please download the tar.gz file [here](https://drive.google.com/file/d/1GmA1Jzqf4RuN7IJUCjInv9IoMcXmJhYO/view?usp=sharing) and unzip to `data/`.

## Dataset Preprocess [Expected: 5-6 hours]

Please run the following scripts which converts raw PCAP/NetFlow data into NetShare-compatible training format.

```Bash
cd preprocess/

# non-privacy data preprocess
bash run_no_privacy.sh

# privacy-related data preprocess
bash run_privacy.sh
```

# Example usage
We provide four different examples to cover the different datasets/scenarios that we described in the paper, i.e., PCAP (CAIDA dataset, with or without differential privacy), NetFlow (UGR16 dataset, with or without differential privacy). These examples use the exact same dataset as in the paper (1 million packets or flow records) but with a much smaller training iterations for quick validation. Therefore, the quality of the results generated by these example configurations (`configs/config_test{i}`) are not informative.

If you would like to reproduce the results in our paper, please use the configuration files named `configs/config{i}` and we recommend to follow the multi-machine setup guideline to achieve the maximum efficiency.

## Prerequisites 
1. You only need a (powerful) single-machine setup to run the four examples here. The estimated time for each step varies on different machines. (We increase the time between different commands so that these examples can be run on even a single machine.
2. Change to super user: `sudo su`
3. Assume that you have followed the [single-machine setup](#single-machine-setup), i.e., creating an Anaconda virtual environment with all dependencies installed. Use `conda activate NetShare` to enter the virtual environment.
4. Assume that you put the `NetShare` repository under `/nfs/NetShare`.
5. **IMPORTANT: FOR EVERY COMMAND, please wait until all python3 processes to finish. You may use `ps aux | grep python3` to check the running python3 processes.**

## Example 1: PCAP without differential privacy
### 1. Input
`data/1M/caida/raw.pcap`

### 2. Training [~10 mins]
```Bash
cd src/
python3 main.py --config_file config_test1_pcap_no_dp --measurer_file measurers_localhost.ini --measurement
```

### 3. Generation [~60 mins]
```Bash
cd src/
python3 main.py --config_file config_test1_pcap_no_dp --measurer_file measurers_localhost.ini --generation
```

### 4. Output
`results/results_test1_pcap_no_dp/1M/caida/split-multiepoch_dep_v2,maxFlowLen-2682,Norm-ZERO_ONE,vecSize-10,df2epochs-fixed_time,interarrival-True,fullIPHdr-True,encodeIP-False/iteration-40,run-0,sample_len-10,pretrain_non_dp-True,pretrain_non_dp_reduce_time-4.0,pretrain_dp-False,dp_noise_multiplier-None,dp_l2_norm_clip-None,pretrain_dir-None/syn_dfs/syn.pcap`

---

## Example 2: PCAP with differential privacy
### 1. Input (private data to be shared)
`data/1M_privacy/caida/raw.csv` (converted from the pcap in Example 1, i.e., `data/1M/caida/raw.pcap`)

### 2. Public data training [~10 mins]
```Bash
cd src/
python3 main.py --config_file config_test_public_pcap --measurer_file measurers_localhost.ini --measurement
```
### 3. Private data training [~40 mins]
**For this example, we only run the first five chunks for sake of resource/time efficiency.**

(The `pretrain_dir` in `config_test2_pcap_dp` has been set to the public data pretrained model checkpoints.)

```Bash
cd src/
python3 main.py --config_file config_test2_pcap_dp --measurer_file measurers_localhost.ini --measurement
```
### 4. Generation [~60 mins]
```Bash
cd src/
python3 main.py --config_file config_test2_pcap_dp --measurer_file measurers_localhost.ini --generation
```

### 5. Output
`results/results_test2_pcap_dp/1M_privacy/caida/split-multiepoch_dep_v2,maxFlowLen-5000,Norm-ZERO_ONE,vecSize-10,df2epochs-fixed_time,interarrival-True,fullIPHdr-True,encodeIP-False/iteration-5,run-0,sample_len-10,pretrain_non_dp-False,pretrain_non_dp_reduce_time-None,pretrain_dp-True,dp_noise_multiplier-0.1,dp_l2_norm_clip-1.0/syn_dfs/syn.pcap`

---

## Example 3: NetFlow without differential privacy
### 1. Input
`data/1M/ugr16/raw.csv`

### 2. Training [~5 mins]
```Bash
cd src/
python3 main.py --config_file config_test3_netflow_no_dp --measurer_file measurers_localhost.ini --measurement
```

### 3. Generation [~40 mins]
```Bash
cd src/
python3 main.py --config_file config_test3_netflow_no_dp --measurer_file measurers_localhost.ini --generation
```

### 4. Ouput
`results/results_test3_netflow_no_dp/1M/ugr16/split-multiepoch_dep_v2,maxFlowLen-34,Norm-ZERO_ONE,vecSize-10,df2epochs-fixed_time,interarrival-True,fullIPHdr-False,encodeIP-False/iteration-40,run-0,sample_len-1,pretrain_non_dp-True,pretrain_non_dp_reduce_time-4.0,pretrain_dp-False,dp_noise_multiplier-None,dp_l2_norm_clip-None,pretrain_dir-None/syn_dfs/syn.csv`

---


## Example 4: NetFlow with differential privacy
### 1. Input (private data to be shared)
`data/1M_privacy/ugr16/raw.csv` (the same csv as Example 3)

### 2. Public data training [~10 mins]
```Bash
cd src/
python3 main.py --config_file config_test_public_netflow --measurer_file measurers_localhost.ini --measurement
```

### 3. Private data Training [~40 mins]
**For this example, we only run the first five chunks for sake of resource/time efficiency.**

```Bash
cd src/
python3 main.py --config_file config_test4_netflow_dp --measurer_file measurers_localhost.ini --measurement
```

### 4. Generation [~40 mins]
```Bash
cd src/
python3 main.py --config_file config_test4_netflow_dp --measurer_file measurers_localhost.ini --generation
```

### 5. Output
`results/results_test4_netflow_dp/1M_privacy/ugr16/split-multiepoch_dep_v2,maxFlowLen-100,Norm-ZERO_ONE,vecSize-10,df2epochs-fixed_time,interarrival-True,fullIPHdr-False,encodeIP-False/iteration-5,run-0,sample_len-1,pretrain_non_dp-False,pretrain_non_dp_reduce_time-None,pretrain_dp-True,dp_noise_multiplier-0.1,dp_l2_norm_clip-1.0/syn_dfs/syn.csv`


# TODOs/Roadmap
- [x] Single-machine version
- [ ] detailed data schema/user manual
- [ ] downstream application
- [ ] Results reproduced scripts
- [ ] Upload checkpoints
- [ ] Corresponding web service in build
- [ ] Pre-packaged docker/AMI
- [ ] Python package

# Refererence
Part of the source code is adapated from the following open-source projects:

- [DoppelGANger](https://github.com/fjxmlzn/DoppelGANger)
- [GPUTaskScheduler](https://github.com/fjxmlzn/GPUTaskScheduler)
- [BSN](https://github.com/fjxmlzn/BSN)