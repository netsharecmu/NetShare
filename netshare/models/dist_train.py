import math
import json
import sys

import numpy as np

from netshare.utils import output

import os

import numpy as np
import torch
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from netshare.models.doppelganger_torch.doppelganger import DoppelGANger  # NOQA
from netshare.models.doppelganger_torch.load_data import load_data  # NOQA
from netshare.models.doppelganger_torch.util import (add_gen_flag, normalize_per_sample,  # NOQA
                      renormalize_per_sample)

# -----------------------------------------------------------------------------
# various inits, derived attributes, I/O setup

# DDP settings
backend = 'nccl'  # 'nccl', 'gloo', etc.
# system
device = 'cuda'  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True  # use PyTorch 2.0 to compile the model to be faster

ddp = int(os.environ.get('RANK', -1)) != -1  # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    # this process will do logging, checkpointing etc.
    master_process = ddp_rank == 0
    seed_offset = ddp_rank  # each process gets a different seed
    print("ddp:", ddp, "ddp_rank:", ddp_rank, "ddp_local_rank:",
      ddp_local_rank, "ddp_world_size:", ddp_world_size, "device:", device)
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
    ddp_local_rank = 0

    print("ddp:", ddp)

if __name__ == "__main__":

    config_path = sys.argv[1]
    with open(config_path, "r") as f:
        _config = json.load(f)

    # load data
    (
        data_feature,
        data_attribute,
        data_gen_flag,
        data_feature_outputs,
        data_attribute_outputs,
    ) = load_data(
        path=_config["dataset"],
        sample_len=_config["sample_len"])
    num_real_attribute = len(data_attribute_outputs)

    # self-norm if applicable
    if _config["self_norm"]:
        (
            data_feature,
            data_attribute,
            data_attribute_outputs,
            real_attribute_mask
        ) = normalize_per_sample(
            data_feature,
            data_attribute,
            data_feature_outputs,
            data_attribute_outputs)
    else:
        real_attribute_mask = [True] * num_real_attribute

    data_feature, data_feature_outputs = add_gen_flag(
        data_feature, data_gen_flag, data_feature_outputs, _config["sample_len"]
    )

    # create directories
    checkpoint_dir = os.path.join(
        _config["result_folder"],
        "checkpoint")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    sample_dir = os.path.join(_config["result_folder"], "sample")
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
    time_path = os.path.join(_config["result_folder"], "time.txt")
    dg = DoppelGANger(
        device=device,
        ddp=ddp,
        ddp_local_rank=ddp_local_rank,
        master_process=master_process,
        ddp_world_size=ddp_world_size,
        checkpoint_dir=checkpoint_dir,
        sample_dir=None,
        time_path=time_path,
        batch_size=_config["batch_size"],
        real_attribute_mask=real_attribute_mask,
        max_sequence_len=data_feature.shape[1],
        sample_len=_config["sample_len"],
        data_feature_outputs=data_feature_outputs,
        data_attribute_outputs=data_attribute_outputs,
        vis_freq=_config["vis_freq"],
        vis_num_sample=_config["vis_num_sample"],
        d_rounds=_config["d_rounds"],
        g_rounds=_config["g_rounds"],
        d_gp_coe=_config["d_gp_coe"],
        num_packing=_config["num_packing"],
        use_attr_discriminator=_config["use_attr_discriminator"],
        attr_d_gp_coe=_config["attr_d_gp_coe"],
        g_attr_d_coe=_config["g_attr_d_coe"],
        epoch_checkpoint_freq=_config["epoch_checkpoint_freq"],
        attribute_latent_dim=_config["attribute_latent_dim"],
        feature_latent_dim=_config["feature_latent_dim"],
        g_lr=_config["g_lr"],
        g_beta1=_config["g_beta1"],
        d_lr=_config["d_lr"],
        d_beta1=_config["d_beta1"],
        attr_d_lr=_config["attr_d_lr"],
        attr_d_beta1=_config["attr_d_beta1"],
        adam_eps=_config["adam_eps"],
        adam_amsgrad=_config["adam_amsgrad"],
        generator_attribute_num_units=_config["generator_attribute_num_units"],
        generator_attribute_num_layers=_config["generator_attribute_num_layers"],
        generator_feature_num_units=_config["generator_feature_num_units"],
        generator_feature_num_layers=_config["generator_feature_num_layers"],
        use_adaptive_rolling=_config["use_adaptive_rolling"],
        discriminator_num_layers=_config["discriminator_num_layers"],
        discriminator_num_units=_config["discriminator_num_units"],
        attr_discriminator_num_layers=_config["attr_discriminator_num_layers"],
        attr_discriminator_num_units=_config["attr_discriminator_num_units"],
        restore=getattr(_config, "restore", False),
        pretrain_dir=_config["pretrain_dir"],
    )
    dg.train(
        epochs=_config["epochs"],
        data_feature=data_feature,
        data_attribute=data_attribute,
        data_gen_flag=data_gen_flag,
    )

    if ddp:
        destroy_process_group()