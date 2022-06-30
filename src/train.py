import os, sys, configparser, json, importlib, math, pickle, copy
import numpy as np
import tensorflow as tf

from gan import output
sys.modules["output"] = output

from gan.doppelganger import DoppelGANger
from gan.util import add_gen_flag, normalize_per_sample
from gan.load_data import load_data
from gan.network import DoppelGANgerGenerator, Discriminator, AttrDiscriminator, RNNInitialStateType
from gan.dataset import NetShareDataset

def trainDG(config):
    print("Currently training with config:", config)

    # save config to the result folder
    with open(os.path.join(config["result_folder"], "config.json"), 'w') as fout:
        json.dump(config, fout)
    
    # load data
    data_in_dir = os.path.join("../", "data", config["dataset"])
    with open(os.path.join(data_in_dir, "data_feature_output.pkl"), "rb") as f:
        data_feature_outputs = pickle.load(f)
    with open(os.path.join(data_in_dir, "data_attribute_output.pkl"), "rb") as f:
        data_attribute_outputs = pickle.load(f)
    dataset = NetShareDataset(
        root=data_in_dir,
        config=config,
        data_attribute_outputs=data_attribute_outputs,
        data_feature_outputs=data_feature_outputs)

    # Run dataset.sample_batch() once to intialize data_attribute_outputs and data_feature_outputs
    dataset.sample_batch(config["batch_size"])
    if (dataset.data_attribute_outputs_train is None) \
        or (dataset.data_feature_outputs_train is None) \
        or (dataset.real_attribute_mask is None):
            print(dataset.data_attribute_outputs_train)
            print(dataset.data_feature_outputs_train)
            print(dataset.real_attribute_mask)
            raise Exception("Dataset variables are not initialized properly for training purposes!")

    sample_len = config["sample_len"]
    data_attribute_outputs = dataset.data_attribute_outputs_train
    data_feature_outputs = dataset.data_feature_outputs_train
    real_attribute_mask = dataset.real_attribute_mask
    gt_lengths = dataset.gt_lengths

    initial_state = None
    if config["initial_state"] == "variable":
        initial_state = RNNInitialStateType.VARIABLE
    elif config["initial_state"] == "random":
        initial_state = RNNInitialStateType.RANDOM
    elif config["initial_state"] == "zero":
        initial_state = RNNInitialStateType.ZERO
    else:
        raise NotImplementedError

    generator = DoppelGANgerGenerator(
        feed_back=config["feed_back"],
        noise=config["noise"],
        attr_noise_type=config["attr_noise_type"],
        feature_noise_type=config["feature_noise_type"],
        feature_outputs=data_feature_outputs,
        attribute_outputs=data_attribute_outputs,
        real_attribute_mask=real_attribute_mask,
        sample_len=sample_len,
        feature_num_layers=config["gen_feature_num_layers"],
        feature_num_units=config["gen_feature_num_units"],
        attribute_num_layers=config["gen_attribute_num_layers"],
        attribute_num_units=config["gen_attribute_num_units"],
        rnn_mlp_num_layers=config["rnn_mlp_num_layers"],
        initial_state=initial_state,
        gt_lengths=gt_lengths,
        use_uniform_lengths=config["use_uniform_lengths"])
    discriminator = Discriminator(
        scale=config["scale"],
        sn_mode=config["sn_mode"],
        num_layers=config["disc_num_layers"],
        num_units=config["disc_num_units"],
        leaky_relu=config["leaky_relu"])
    if config["aux_disc"]:
        attr_discriminator = AttrDiscriminator(
            scale=config["scale"],
            sn_mode=config["sn_mode"],
            num_layers=config["attr_disc_num_layers"],
            num_units=config["attr_disc_num_units"],
            leaky_relu=config["leaky_relu"])


    checkpoint_dir = os.path.join(config["result_folder"], "checkpoint") 
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    sample_dir = os.path.join(config["result_folder"], "sample")
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    time_path = os.path.join(config["result_folder"], "time.txt")


    # run_config = tf.ConfigProto()

    if config["num_cores"] == None:
        run_config = tf.ConfigProto()
    else:
        num_cores = config["num_cores"] # it means number of cores
        run_config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,
                            inter_op_parallelism_threads=num_cores,
                            allow_soft_placement=True,
                            device_count={'CPU': num_cores})

    with tf.Session(config=run_config) as sess:
        gan = DoppelGANger(
            sess=sess,
            checkpoint_dir=checkpoint_dir,
            pretrain_dir=config["pretrain_dir"],
            sample_dir=sample_dir,
            time_path=time_path,
            batch_size=config["batch_size"],
            iteration=config["iteration"],
            dataset=dataset,
            sample_len=sample_len,
            real_attribute_mask=real_attribute_mask,
            data_feature_outputs=data_feature_outputs,
            data_attribute_outputs=data_attribute_outputs,
            vis_freq=config["vis_freq"],
            vis_num_sample=config["vis_num_sample"],
            generator=generator,
            discriminator=discriminator,
            attr_discriminator=(attr_discriminator
                                if config["aux_disc"] else None),
            d_gp_coe=config["d_gp_coe"],
            attr_d_gp_coe=(config["attr_d_gp_coe"]
                           if config["aux_disc"] else 0.0),
            g_attr_d_coe=(config["g_attr_d_coe"]
                          if config["aux_disc"] else 0.0),
            d_rounds=config["d_rounds"],
            g_rounds=config["g_rounds"],
            fix_feature_network=config["fix_feature_network"],
            g_lr=config["g_lr"],
            d_lr=config["d_lr"],
            attr_d_lr=(config["attr_d_lr"]
                       if config["aux_disc"] else 0.0),
            extra_checkpoint_freq=config["extra_checkpoint_freq"],
            epoch_checkpoint_freq=config["epoch_checkpoint_freq"],
            num_packing=config["num_packing"],

            debug=config["debug"],
            combined_disc=config["combined_disc"],

            # DP-related
            dp_noise_multiplier=config["dp_noise_multiplier"],
            dp_l2_norm_clip=config["dp_l2_norm_clip"],

            # SN-related
            sn_mode=config["sn_mode"]
            )

        gan.build()
        # print("restore_flag:", restore_flag)
        gan.train(restore=config["restore"])

    dataset.stop_data_loader()


def main():
    config_json_file = sys.argv[1]
    config_idx = int(sys.argv[2])
    
    fin = open(config_json_file, "r")
    configs = json.load(fin)
    fin.close()

    trainDG(configs[config_idx])


if __name__ == "__main__":
    main()