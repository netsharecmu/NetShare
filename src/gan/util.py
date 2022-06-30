import sys, configparser, json, subprocess, time, argparse, datetime
import importlib
import os, re, copy, random, warnings
import tensorflow as tf
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# https://stackoverflow.com/questions/5228158/cartesian-product-of-a-dictionary-of-lists
import itertools

from .output import OutputType, Output, Normalization
from tqdm import tqdm
from multiprocessing import Process

try:
    from tensorflow_privacy.privacy.optimizers import dp_optimizer
    from tensorflow_privacy.privacy.analysis.compute_dp_sgd_privacy_lib import compute_dp_sgd_privacy
    from tensorflow_privacy.privacy.analysis.rdp_accountant import compute_rdp
    from tensorflow_privacy.privacy.analysis.rdp_accountant import get_privacy_spent
except:
    pass

def draw_attribute(data, outputs, path=None):
    if isinstance(data, list):
        num_sample = len(data)
    else:
        num_sample = data.shape[0]
    id_ = 0
    for i in range(len(outputs)):
        if outputs[i].type_ == OutputType.CONTINUOUS:
            for j in range(outputs[i].dim):
                plt.figure()
                for k in range(num_sample):
                    plt.scatter(
                        k,
                        data[k][id_],
                        s=12)
                if path is None:
                    plt.show()
                else:
                    plt.savefig("{},output-{},dim-{}.png".format(path, i, j))
                plt.xlabel("sample")
                plt.close()
                id_ += 1
        elif outputs[i].type_ == OutputType.DISCRETE:
            plt.figure()
            for j in range(num_sample):
                plt.scatter(
                    j,
                    np.argmax(data[j][id_: id_ + outputs[i].dim],
                              axis=0),
                    s=12)
            plt.xlabel("sample")
            if path is None:
                plt.show()
            else:
                plt.savefig("{},output-{}.png".format(path, i))
            plt.close()
            id_ += outputs[i].dim
        else:
            raise Exception("unknown output type")


def draw_feature(data, lengths, outputs, path=None):
    if isinstance(data, list):
        num_sample = len(data)
    else:
        num_sample = data.shape[0]
    id_ = 0
    for i in range(len(outputs)):
        if outputs[i].type_ == OutputType.CONTINUOUS:
            for j in range(outputs[i].dim):
                plt.figure()
                for k in range(num_sample):
                    plt.plot(
                        range(int(lengths[k])),
                        data[k][:int(lengths[k]), id_],
                        "o-",
                        markersize=3,
                        label="sample-{}".format(k))
                plt.legend()
                if path is None:
                    plt.show()
                else:
                    plt.savefig("{},output-{},dim-{}.png".format(path, i, j))
                plt.close()
                id_ += 1
        elif outputs[i].type_ == OutputType.DISCRETE:
            plt.figure()
            for j in range(num_sample):
                plt.plot(
                    range(int(lengths[j])),
                    np.argmax(data[j][:int(lengths[j]),
                                      id_: id_ + outputs[i].dim],
                              axis=1),
                    "o-",
                    markersize=3,
                    label="sample-{}".format(j))

            plt.legend()
            if path is None:
                plt.show()
            else:
                plt.savefig("{},output-{}.png".format(path, i))
            plt.close()
            id_ += outputs[i].dim
        else:
            raise Exception("unknown output type")


def renormalize_per_sample(data_feature, data_attribute, data_feature_outputs,
                           data_attribute_outputs, gen_flags,
                           num_real_attribute):
    attr_dim = 0
    for i in range(num_real_attribute):
        attr_dim += data_attribute_outputs[i].dim
    attr_dim_cp = attr_dim

    fea_dim = 0
    for output in data_feature_outputs:
        if output.type_ == OutputType.CONTINUOUS:
            for _ in range(output.dim):
                max_plus_min_d_2 = data_attribute[:, attr_dim]
                max_minus_min_d_2 = data_attribute[:, attr_dim + 1]
                attr_dim += 2

                max_ = max_plus_min_d_2 + max_minus_min_d_2
                min_ = max_plus_min_d_2 - max_minus_min_d_2

                max_ = np.expand_dims(max_, axis=1)
                min_ = np.expand_dims(min_, axis=1)

                if output.normalization == Normalization.MINUSONE_ONE:
                    data_feature[:, :, fea_dim] = \
                        (data_feature[:, :, fea_dim] + 1.0) / 2.0

                data_feature[:, :, fea_dim] = \
                    data_feature[:, :, fea_dim] * (max_ - min_) + min_

                fea_dim += 1
        else:
            fea_dim += output.dim

    tmp_gen_flags = np.expand_dims(gen_flags, axis=2)
    data_feature = data_feature * tmp_gen_flags

    data_attribute = data_attribute[:, 0: attr_dim_cp]

    return data_feature, data_attribute


def normalize_per_sample(data_feature, data_attribute, data_gen_flag, 
                    data_feature_outputs, data_attribute_outputs, eps=1e-4):
    data_feature_outputs = copy.deepcopy(data_feature_outputs)
    data_attribute_outputs = copy.deepcopy(data_attribute_outputs)

    # print("\n\nnormalizing per sample...")
    data_feature_min = np.amin(data_feature, axis=1)
    data_feature_max = np.amax(data_feature, axis=1)
    # print("Before masking...")
    # print("data_feature_avg:", np.average(data_feature, axis=1))
    # print("data_feature_min_avg:", np.average(data_feature_min, axis=1))
    # print("data_feature_max_avg:", np.average(data_feature_max, axis=1))

    # remove padded values before fetching per-sample min/max
    # assume all samples have maximum length
    data_feature_min = np.empty_like(data_feature_min)
    data_feature_max = np.empty_like(data_feature_max)
    sample_length = np.count_nonzero(data_gen_flag, axis=1)

    # iterate over samples
    for i in range(data_feature.shape[0]):
        for k in range(data_feature.shape[2]):
            data_feature_min[i][k] = np.min(data_feature[i, :sample_length[i], k])
            data_feature_max[i][k] = np.max(data_feature[i, :sample_length[i], k])
    
    # print("After masking...")
    # print("data_feature_min_avg:", np.average(data_feature_min, axis=1))
    # print("data_feature_max_avg:", np.average(data_feature_max, axis=1))

    additional_attribute = []
    additional_attribute_outputs = []

    dim = 0
    for output in data_feature_outputs:
        if output.type_ == OutputType.CONTINUOUS:
            for _ in range(output.dim):
                max_ = data_feature_max[:, dim] + eps
                min_ = data_feature_min[:, dim] - eps

                additional_attribute.append((max_ + min_) / 2.0)
                additional_attribute.append((max_ - min_) / 2.0)
                additional_attribute_outputs.append(Output(
                    type_=OutputType.CONTINUOUS,
                    dim=1,
                    normalization=output.normalization,
                    is_gen_flag=False))
                additional_attribute_outputs.append(Output(
                    type_=OutputType.CONTINUOUS,
                    dim=1,
                    normalization=Normalization.ZERO_ONE,
                    is_gen_flag=False))

                max_ = np.expand_dims(max_, axis=1)
                min_ = np.expand_dims(min_, axis=1)

                data_feature[:, :, dim] = \
                    (data_feature[:, :, dim] - min_) / (max_ - min_)
                if output.normalization == Normalization.MINUSONE_ONE:
                    data_feature[:, :, dim] = \
                        data_feature[:, :, dim] * 2.0 - 1.0

                dim += 1
        else:
            dim += output.dim
    
    # After normalization, force padded values to be zero
    for i in range(data_feature.shape[0]):
        for j in range(data_feature.shape[1]):
            if data_gen_flag[i][j] == 0.0:
                data_feature[i, j, :] = 0.0
    # print("data_feature_avg:", np.average(data_feature, axis=1))


    real_attribute_mask = ([True] * len(data_attribute_outputs) +
                           [False] * len(additional_attribute_outputs))

    additional_attribute = np.stack(additional_attribute, axis=1)
    data_attribute = np.concatenate(
        [data_attribute, additional_attribute], axis=1)
    data_attribute_outputs.extend(additional_attribute_outputs)

    return data_feature, data_attribute, data_attribute_outputs, \
        real_attribute_mask


def add_gen_flag(data_feature, data_gen_flag, data_feature_outputs,
                 sample_len):
    data_feature_outputs = copy.deepcopy(data_feature_outputs)
    
    for output in data_feature_outputs:
        if output.is_gen_flag:
            raise Exception("is_gen_flag should be False for all"
                            "feature_outputs")

    if (data_feature.shape[2] !=
            np.sum([t.dim for t in data_feature_outputs])):
        raise Exception("feature dimension does not match feature_outputs")

    if len(data_gen_flag.shape) != 2:
        raise Exception("data_gen_flag should be 2 dimension")

    num_sample, length = data_gen_flag.shape

    data_gen_flag = np.expand_dims(data_gen_flag, 2)

    data_feature_outputs.append(Output(
        type_=OutputType.DISCRETE,
        dim=2,
        is_gen_flag=True))

    shift_gen_flag = np.concatenate(
        [data_gen_flag[:, 1:, :],
         np.zeros((data_gen_flag.shape[0], 1, 1))],
        axis=1)
    if length % sample_len != 0:
        raise Exception("length must be a multiple of sample_len")
    data_gen_flag_t = np.reshape(
        data_gen_flag,
        [num_sample, int(length / sample_len), sample_len])
    data_gen_flag_t = np.sum(data_gen_flag_t, 2)
    data_gen_flag_t = data_gen_flag_t > 0.5
    data_gen_flag_t = np.repeat(data_gen_flag_t, sample_len, axis=1)
    data_gen_flag_t = np.expand_dims(data_gen_flag_t, 2)
    data_feature = np.concatenate(
        [data_feature,
         shift_gen_flag,
         (1 - shift_gen_flag) * data_gen_flag_t],
        axis=2)

    return data_feature, data_feature_outputs


def append_data_feature(data_feature, max_flow_len):
    feature_dim = len(data_feature[0][0])
    new_data_feature = []
    for row in tqdm(data_feature):
        new_row = list(copy.deepcopy(row))
        for i in range(max_flow_len - len(row)):
            new_row.append([0.0]*feature_dim)
        new_data_feature.append(new_row)

    return np.asarray(new_data_feature)

def append_data_gen_flag(data_gen_flag, max_flow_len):
    new_data_gen_flag = []
    for row in tqdm(data_gen_flag):
        new_row = list(copy.deepcopy(row))
        for i in range(max_flow_len - len(row)):
            new_row.append(0.0)
        new_data_gen_flag.append(new_row)

    return np.asarray(new_data_gen_flag)

def compute_dp_sgd_privacy_single(
        n_samples,
        batch_size, 
        n_iterations,
        noise_multiplier,
        delta
    ):

    """Compute epsilon based on the given hyperparameters."""
    q = batch_size / n_samples  # q - the sampling ratio.
    if q > 1:
        raise ValueError('n must be larger than the batch size.')
    orders = ([1.25, 1.5, 1.75, 2., 2.25, 2.5, 3., 3.5, 4., 4.5] +
                list(range(5, 64)) + [128, 256, 512])
    steps = n_iterations
    rdp = compute_rdp(q, noise_multiplier, steps, orders)

    return rdp


# compute the total (epsilon, delta) budget, i.e., training + generation
# tensorflow-privacy==0.5.0 (latest version compatible with tf=1.15)
# TODO: bump to latest tf-privacy version
def compute_dp_sgd_privacy_sum(
        n_samples_list, # len(n_samples_list) == n_chunks
        batch_size,
        n_iterations, 
        noise_multiplier_train, # for training
        noise_multiplier_gen, # for generation, i.e., estimating flow length 
        delta
    ):
    
    rdp_sum_train = 0
    # privacy budget for training
    for chunkid, n_samples in enumerate(n_samples_list):
        rdp = compute_dp_sgd_privacy_single(
            n_samples=n_samples,
            batch_size=batch_size, 
            n_iterations=n_iterations,
            noise_multiplier=noise_multiplier_train,
            delta=delta
        )
        rdp_sum_train += rdp
    
    rdp_sum_gen = 0
    # privacy budget for generation (flow size estimation)
    for chunkid, n_samples in enumerate(n_samples_list):
        rdp = compute_dp_sgd_privacy_single(
            n_samples=1,
            batch_size=1, 
            n_iterations=1,
            noise_multiplier=noise_multiplier_gen,
            delta=delta
        )
        rdp_sum_gen += rdp
    
    # convert RDP to DP
    orders = ([1.25, 1.5, 1.75, 2., 2.25, 2.5, 3., 3.5, 4., 4.5] +
                list(range(5, 64)) + [128, 256, 512])
    eps, _, opt_order = get_privacy_spent(orders, rdp_sum_train+rdp_sum_gen, target_delta=delta)

    # print("EPS training: {}, generation: {}".format(
    #     get_privacy_spent(orders, rdp_sum_train, target_delta=delta)[0],
    #     get_privacy_spent(orders, rdp_sum_gen, target_delta=delta)[0]
    # ))

    return eps, delta

def estimate_flowlen_dp(
        n_samples_list, # len(n_samples_list) == n_chunks
        noise_multiplier_gen=1.0,
        seed=42
    ):
    np.random.seed(seed)
    flowlen_list_dp = []
    for chunkid, n_samples in enumerate(n_samples_list):
        flowlen_list_dp.append(n_samples + np.random.normal(0, noise_multiplier_gen))
        
    return flowlen_list_dp

def dict_product(dicts):
    return (dict(zip(dicts.keys(), x)) for x in itertools.product(*dicts.values()))

def chunks(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

# return a list of configs
def load_config(config):
    configs = []

    for i in range(len(config["test_config"])):
        for test_config in list(dict_product(config["test_config"][i])):

            config_ = {**config["scheduler_config"], **config["global_config"], **test_config}
            
            sub_result_folder = ",".join("{}-{}".format(str(k), str(v)) for k, v in test_config.items() if k not in (config["scheduler_config"]["ignored_keys_for_folder_name"]))

            config_["result_folder"] = os.path.join(config_["result_root_folder"], test_config["dataset"], sub_result_folder)
            
            # sanity check
            if config_["pretrain_non_dp"] == True and ((config_["dp_noise_multiplier"] is not None) or (config_["dp_l2_norm_clip"] is not None)):
                raise ValueError("pretrain_non_DP can only be used for non-DP case!")

            if config_["pretrain_non_dp"] == True and config_["pretrain_non_dp_reduce_time"] is None:
                raise ValueError("pretrain_non_dp=True, then pretrain_non_dp_reduce_time must be set!")

            if config_["pretrain_non_dp"] == False and config_["pretrain_non_dp_reduce_time"] is not None:
                raise ValueError("pretrain_non_dp=False, pretrain_non_dp_reduce_time does not need to be set!")

            if config_["pretrain_non_dp"] == True and config_["pretrain_dp"] == True:
                raise ValueError("Only one of pretrain_non_DP and pretrain_DP can be True!")
            
            if config_["pretrain_dp"] == True and config_["pretrain_dir"] is None:
                raise ValueError("You are using DP with pretrained public model, pretrain_dir must be set to the pretrained public model checkpoint directory!")
            
            configs.append(config_)

    return configs

def get_configid_from_kv(configs, k, v):
    for idx, config in enumerate(configs):
        if config[k] == v:
            return idx
    raise ValueError("{}: {} not found in configs!".format(k, v))


# convert a list of configurations to a grouped dictionary for training purpose
# key : value
# dp : bool
# pretrain: bool
# config_ids: list
def configs2configsgroup(configs, generation_flag=False):
    config_id_list_victim = [i for i in range(len(configs))]
    config_group_list = []

    for config_id, config in enumerate(configs):
        if config_id in config_id_list_victim:
            config_group = {}
            config_group["dataset"] = "/".join(config["dataset"].split("/")[:2])
            config_group["dp_noise_multiplier"] = config["dp_noise_multiplier"]
            config_group["dp"] = (config["dp_noise_multiplier"] is not None)
            config_group["pretrain"] = (config["pretrain_non_dp"] or config["pretrain_dp"])
            config_group["config_ids"] = []
            
            num_chunks = config["num_chunks"]
            for chunk_idx in range(num_chunks):
                config_id_ = get_configid_from_kv(configs, "result_folder", re.sub('epochid-[0-9]+', 'epochid-{}'.format(chunk_idx), config["result_folder"]))
                config_group["config_ids"].append(config_id_)
                config_id_list_victim.remove(config_id_)

            
            config_group_list.append(config_group)
        
    # sanity check
    assert len(config_id_list_victim) == 0

    config_ids_check = []
    for config_group in config_group_list:
        config_ids_check += config_group["config_ids"]
        print(config_group)
    assert set(config_ids_check) == set([i for i in range(len(configs))])

    # add pretrain_dir etc. to the original configs
    for config_group in config_group_list:
        if config_group["pretrain"] == False:
            for config_id in config_group["config_ids"]:
                configs[config_id]["restore"] = False
        else:
            if config_group["dp"] == False:
                for chunk_id, config_id in enumerate(config_group["config_ids"]):
                    if chunk_id == 0:
                        chunk0_idx = config_id
                        configs[config_id]["restore"] = False
                        iteration_range = list(range(configs[config_id]["extra_checkpoint_freq"] - 1,
                              configs[config_id]["iteration"],
                              configs[config_id]["extra_checkpoint_freq"]))
                        iteration_range.reverse()
                        
                        pretrain_dir = None
                        # use last available ckpt
                        if configs[config_id]["skip_chunk0_train"] == True:
                            last_iteration_found = False
                            for iteration_id in iteration_range:
                                if last_iteration_found == True:
                                    break
                                ckpt_dir = os.path.join(
                                    configs[config_id]["result_folder"],
                                    "checkpoint",
                                    "iteration_id-{}".format(iteration_id)
                                )
                                if os.path.exists(ckpt_dir):
                                    last_iteration_found = True
                            
                            if last_iteration_found == False:
                                warnings.warn("Skipping chunk0 training but chunk0 has no available ckpt at {}! Please move ckpts into the corresponding folder.".format(configs[config_id]["result_folder"]))
                            else:
                                pretrain_dir = ckpt_dir
                        else:
                            if os.path.exists(os.path.join(
                                configs[config_id]["result_folder"],
                                "checkpoint")
                            ) and generation_flag == False:
                                raise ValueError("Chunk0 training NOT skipped but ckpts already exist! Please clean out the ckpt folder to continute training from scratch.")

                            pretrain_dir = os.path.join(
                                configs[config_id]["result_folder"],
                                "checkpoint",
                                "iteration_id-{}".format(configs[config_id]["iteration"]-1)
                            )

                        configs[config_id]["pretrain_dir"] = pretrain_dir

                    else:
                        configs[config_id]["restore"] = True
                        configs[config_id]["pretrain_dir"] = pretrain_dir
                        configs[config_id]["iteration"] = int(configs[config_id]["iteration"] / configs[config_id]["pretrain_non_dp_reduce_time"])

            else:
                for chunk_id, config_id in enumerate(config_group["config_ids"]):
                    configs[config_id]["restore"] = True
    
    # add chunk_id and eval_root_folder for generation related
    for config_group in config_group_list:
        chunk0_idx = config_group["config_ids"][0]
        eval_root_folder = re.sub('epochid-0,', '', configs[chunk0_idx]["result_folder"])
        for chunk_id, config_id in enumerate(config_group["config_ids"]):
            configs[config_id]["chunk_id"] = chunk_id
            configs[config_id]["eval_root_folder"] = eval_root_folder
    
    return configs, config_group_list

# multi-process target function:
# ONLY USED FOR dp=False and pretrain=True
# 1. wait for chunk0 to finish
# 2. launch the processes of remaining chunks
# 
# Input:
#   `configs`: a list of config (global)
#   `config_ids`: a group of config_ids for the same dataset/parameters
#               len(config_ids) == number of chunks
#   `dict_configIdx_measureIP`: config_id : measurer_ip
#   `check_frequency`: interval (seconds) to check whether chunk0 is finished
def wait_for_chunk0(config_group_id, configs, config_ids, dict_configIdx_measureIP, sub_python_file, config_json_file):

    while True:
        chunk0_idx = config_ids[0]
        if configs[chunk0_idx]["skip_chunk0_train"] and configs[chunk0_idx]["pretrain_dir"] is None:
            raise ValueError("Skipping chunk0 training but chunk0 has no available ckpt! Please move ckpts into the corresponding folder.")

        if not os.path.exists(configs[chunk0_idx]["pretrain_dir"]):
            print("Chunk0 from config_group_id {} training not finished!".format(config_group_id))
        else:
            time.sleep(configs[chunk0_idx]["sleep_time_check_finish"])
            print("Chunk0 from config_group_id {} training finished!".format(config_group_id))
            print("Start launching experiments for remaining chunks for config_group_id {}...".format(config_group_id))
            for config_idx in config_ids[1:]:
                # sanity check
                if not os.path.exists(configs[config_idx]["pretrain_dir"]):
                    raise ValueError("Pretrain_dir {} does not exist!")
                measurer_ip = dict_configIdx_measureIP[config_idx][1]
                log_file = os.path.join(configs[config_idx]["result_folder"], "worker_train.log")

                cmd = "sudo ssh -o StrictHostKeyChecking=no root@{} \"source ~/anaconda3/etc/profile.d/conda.sh && conda activate {} && cd {} &&  python3 {} {} {} \" > {} 2>&1 &"
                cmd = cmd.format(measurer_ip, configs[config_idx]["conda_virtual_env"], configs[config_idx]["src_dir"], sub_python_file, config_json_file, config_idx, log_file)
                print(cmd)

                subprocess.Popen(cmd, stderr=sys.stdout.fileno(), shell=True)
                time.sleep(configs[config_idx]["sleep_time_launch_cmd"])
            break

        time.sleep(configs[chunk0_idx]["sleep_time_check_finish"])

def load_measurers(path):
    measurer_data = configparser.ConfigParser()
    measurer_data.optionxform = str
    measurer_data.read(path)

    numMeasurers = int(measurer_data["measurer"]["numMeasurers"])

    measurer_IPs = []
    for i in range(numMeasurers):
        if "Measurer_{}".format(i+1) in measurer_data:
            measurer_IPs.append(measurer_data["Measurer_"+str(i+1)]["ip"])

    # print("# of measurers:", len(measurer_IPs))

    return measurer_IPs