#!/bin/bash
cd $HOME

VIRTUAL_ENV=$1
USERNAME=$2
CONDA_EXEC=$HOME/anaconda3/bin/conda
NETSHARE_LOCAL_REPO=/nfs/NetShare

# Anaconda3
if [ -f $CONDA_EXEC ] 
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

# If already cloned
if ! [ -d $NETSHARE_LOCAL_REPO]
then
    echo "git clone from remote repo..."
    git clone https://github.com/netsharecmu/NetShare.git $NETSHARE_LOCAL_REPO
else
    echo "$NETSHARE_LOCAL_REPO exists! Skip git clone..."
fi

export LC_ALL=C.UTF-8
export LANG=C.UTF-8
ray start --head && ray stop

cd $NETSHARE_LOCAL_REPO
pip3 install -e .