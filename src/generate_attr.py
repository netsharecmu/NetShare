# generate attributes only
import os, sys, configparser, json, importlib, math, pickle
import tensorflow as tf
import numpy as np

from gan import output
sys.modules["output"] = output

from gan.doppelganger import DoppelGANger
from gan.util import add_gen_flag, normalize_per_sample, renormalize_per_sample, append_data_feature, append_data_gen_flag
from gan.load_data import load_data
from gan.network import DoppelGANgerGenerator, Discriminator, AttrDiscriminator, RNNInitialStateType
from gan.dataset import NetShareDataset

from gan.util import estimate_flowlen_dp

generatedSamples_per_epoch = 1

def closest(lst, K):
     lst = np.asarray(lst)
     idx = (np.abs(lst - K)).argmin()
     return lst[idx]

def trainDG(config):
    if "/caida/" in config["dataset"] or "/dc/" in config["dataset"] or "/ca/" in config["dataset"]:
        data_type = "pcap"
    elif "ugr16" in config["dataset"] or "cidds" in config["dataset"] or "ton" in config["dataset"]:
        data_type = "netflow"
    else:
        raise ValueError("Unknown data type! Must be netflow or pcap...")

    data_in_dir = os.path.join("../", "data", config["dataset"])
    with open(os.path.join(data_in_dir, "data_feature_output.pkl"), "rb") as f:
        data_feature_outputs = pickle.load(f)
    with open(os.path.join(data_in_dir, "data_attribute_output.pkl"), "rb") as f:
        data_attribute_outputs = pickle.load(f)

    num_real_attribute = len(data_attribute_outputs)

    num_real_samples = len([file for file in os.listdir(os.path.join(data_in_dir, "data_train_npz")) if file.endswith(".npz")])

    # add noise for DP generation
    if config["dp_noise_multiplier"] is not None:
        print("DP case: adding noise to # of flows")
        num_real_samples += int(estimate_flowlen_dp([num_real_samples])[0])
    
    print("orig num_real_samples:", num_real_samples)

    if data_type == "netflow":
        config["generate_num_train_sample"] = int(1.25*num_real_samples)
        config["generate_num_test_sample"] = 0
    elif data_type == "pcap":
        # if "/ca/" in config["dataset"]:
        #     config["generate_num_train_sample"] = int(1.5*num_real_samples)
        #     config["generate_num_test_sample"] = 0
        # else:
        config["generate_num_train_sample"] = num_real_samples
        config["generate_num_test_sample"] = 0

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
        print("Finished building")

        total_generate_num_sample = \
                (config["generate_num_train_sample"] +
                 config["generate_num_test_sample"])
                
        print("total generated sample:", total_generate_num_sample)

        batch_data_attribute, batch_data_feature, batch_data_gen_flag = dataset.sample_batch(config["batch_size"])
        if batch_data_feature.shape[1] % sample_len != 0:
            raise Exception("length must be a multiple of sample_len")
        length = int(batch_data_feature.shape[1] / sample_len)
        real_attribute_input_noise = gan.gen_attribute_input_noise(
            total_generate_num_sample)
        addi_attribute_input_noise = gan.gen_attribute_input_noise(
            total_generate_num_sample)
        feature_input_noise = gan.gen_feature_input_noise(
            total_generate_num_sample, length)
        input_data = gan.gen_feature_input_data_free(
            total_generate_num_sample)

        last_iteration_found = False
        iteration_range = list(range(config["extra_checkpoint_freq"] - 1,
                              config["iteration"],
                              config["extra_checkpoint_freq"]))
        # reverse list in place
        iteration_range.reverse()

        # HACKY EXPERIMENTAL FEATURE; TO BE REVERTED
        # if config["chunk_id"] == 0:
        #     TARGET_ITERATION_ID = int(num_real_samples / config["batch_size"] * 400)
        # else:
        #     TARGET_ITERATION_ID = int(num_real_samples / config["batch_size"] * 100)
        # CLOSEST_ITERATION_ID = closest(iteration_range, TARGET_ITERATION_ID)
        # print("Closest iteration id:", CLOSEST_ITERATION_ID)

        for iteration_id in iteration_range:
        # for iteration_id in [CLOSEST_ITERATION_ID]:
            if last_iteration_found == True:
                break
            
            print("Processing iteration_id: {}".format(iteration_id))
            mid_checkpoint_dir = os.path.join(
                checkpoint_dir, "iteration_id-{}".format(iteration_id))
            if not os.path.exists(mid_checkpoint_dir):
                print("Not found {}".format(mid_checkpoint_dir))
                continue
            else:
                last_iteration_found = True

            for generated_samples_idx in range(generatedSamples_per_epoch):
                print("generate {}-th sample from iteration_id-{}".format(generated_samples_idx+1, iteration_id))

                save_path = os.path.join(
                    config["result_folder"],
                    "generated_samples",
                    "iteration_id-{}".format(iteration_id),
                    "sample-{}".format(generated_samples_idx+1))
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                train_path_ori = os.path.join(
                    save_path, "generated_data_train_ori.npz")
                test_path_ori = os.path.join(
                    save_path, "generated_data_test_ori.npz")
                train_path = os.path.join(
                    save_path, "generated_data_train.npz")
                test_path = os.path.join(
                    save_path, "generated_data_test.npz")
                # if os.path.exists(test_path):
                #     print("Save_path {} exists".format(save_path))
                #     continue

                gan.load(mid_checkpoint_dir)

                print("Finished loading")

                features, attributes, gen_flags, lengths = gan.sample_from(
                    real_attribute_input_noise, addi_attribute_input_noise,
                    feature_input_noise, input_data)

                

                # features, attributes, gen_flags, lengths = gan.sample_from(
                #     real_attribute_input_noise, addi_attribute_input_noise,
                #     feature_input_noise, input_data, given_attribute=given_data_attribute)

                # specify given_attribute parameter, if you want to generate
                # data according to an attribute
                print(features.shape)
                print(attributes.shape)
                print(gen_flags.shape)
                print(lengths.shape)

                split = config["generate_num_train_sample"]

                if config["self_norm"]:
                    # np.savez(
                    #     train_path_ori,
                    #     data_feature=features[0: split],
                    #     data_attribute=attributes[0: split],
                    #     data_gen_flag=gen_flags[0: split])
                    # np.savez(
                    #     test_path_ori,
                    #     data_feature=features[split:],
                    #     data_attribute=attributes[split:],
                    #     data_gen_flag=gen_flags[split:])

                    features, attributes = renormalize_per_sample(
                        features, attributes, data_feature_outputs,
                        data_attribute_outputs, gen_flags,
                        num_real_attribute=num_real_attribute)

                    print(features.shape)
                    print(attributes.shape)
                
                save_path = os.path.join(config["eval_root_folder"], "attr_raw")
                os.makedirs(save_path, exist_ok=True)
                np.savez(
                    os.path.join(save_path, "chunk_id-{}.npz".format(config["chunk_id"])),
                    data_attribute=attributes[0:split]
                )

                print("Number of packets this chunk:", np.sum(gen_flags))

        print("Done")
    
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