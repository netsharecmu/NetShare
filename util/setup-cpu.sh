#!/bin/bash
cd $HOME

VIRTUAL_ENV=$1
CONDA_DIR=$HOME/anaconda3/bin/conda

# Anaconda3
if [ -d $CONDA_DIR ] 
then
    echo "Anaconda3 installed."
else
    echo "Anaconda3 not installed. Start installation now..."
    wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh
    bash Anaconda3-2022.05-Linux-x86_64.sh -b -p $HOME/anaconda3
fi
eval "$($HOME/anaconda3/bin/conda shell.bash hook)"
conda init

# create virtual environment if not exists
if ! { conda env list | grep $VIRTUAL_ENV; } >/dev/null 2>&1
then
    echo "Conda environment $VIRTUAL_ENV not installed."
    conda create -y --name $VIRTUAL_ENV python=3.6
else
    echo "Conda environment $VIRTUAL_ENV installed."
fi
source $HOME/anaconda3/etc/profile.d/conda.sh
conda activate $VIRTUAL_ENV

pip3 install tensorflow==1.15
# 0.5.1 won't work for tf==1.15 as tf-estimator is not compatible
# 0.4.0 won't work since it does not have compute API compute_dp_sgd_privacy
# 0.5.0 is compatible with tf==1.15 and will not cause conflict
pip3 install tensorflow-privacy==0.5.0

pip3 install tqdm matplotlib pandas sklearn more-itertools gensim==3.8.3 torch torchvision networkx notebook ipyplot jupyterlab statsmodels gdown annoy pyshark scapy ray "ray[default]" multiprocess addict config_io
