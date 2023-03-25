# type: ignore

import inspect
import json
import os
import pickle

import numpy as np
import tensorflow as tf

import netshare.utils.ray as ray
from netshare.configs import get_config
from netshare.models.doppelganger_tf.dataset import NetShareDataset
from netshare.models.doppelganger_tf.doppelganger import DoppelGANger
from netshare.models.doppelganger_tf.network import (
    AttrDiscriminator,
    Discriminator,
    DoppelGANgerGenerator,
    RNNInitialStateType,
)
from netshare.models.doppelganger_tf.util import (
    estimate_flowlen_dp,
    renormalize_per_sample,
)
from netshare.models.model import Model
from netshare.utils.logger import logger


class DoppelGANgerTFModel(Model):
    def _train(self, input_train_data_folder, output_model_folder, log_folder):
        logger.debug(f"{self.__class__.__name__}.{inspect.stack()[0][3]}")

        self._config["result_folder"] = self._config.get(
            "result_folder", output_model_folder
        )
        self._config["dataset"] = self._config.get("dataset", input_train_data_folder)

        # If Ray is disabled, reset TF graph
        if not ray.config.enabled:
            tf.compat.v1.reset_default_graph()

        # save config to the result folder
        with open(
            os.path.join(self._config["result_folder"], "config.json"), "w"
        ) as fout:
            json.dump(self._config, fout)

        # load data
        data_in_dir = self._config["dataset"]

        with open(os.path.join(data_in_dir, "data_feature_output.pkl"), "rb") as f:
            data_feature_outputs = pickle.load(f)
        with open(os.path.join(data_in_dir, "data_attribute_output.pkl"), "rb") as f:
            data_attribute_outputs = pickle.load(f)

        dataset = NetShareDataset(
            root=data_in_dir,
            config=self._config,
            data_attribute_outputs=data_attribute_outputs,
            data_feature_outputs=data_feature_outputs,
        )

        # Run dataset.sample_batch() once to intialize data_attribute_outputs
        # and data_feature_outputs
        dataset.sample_batch(self._config["batch_size"])
        if (
            (dataset.data_attribute_outputs_train is None)
            or (dataset.data_feature_outputs_train is None)
            or (dataset.real_attribute_mask is None)
        ):
            raise Exception(
                "Dataset variables are not initialized "
                "properly for training purposes!"
            )

        sample_len = self._config["sample_len"]
        data_attribute_outputs = dataset.data_attribute_outputs_train
        data_feature_outputs = dataset.data_feature_outputs_train
        real_attribute_mask = dataset.real_attribute_mask
        gt_lengths = dataset.gt_lengths

        initial_state = None
        if self._config["initial_state"] == "variable":
            initial_state = RNNInitialStateType.VARIABLE
        elif self._config["initial_state"] == "random":
            initial_state = RNNInitialStateType.RANDOM
        elif self._config["initial_state"] == "zero":
            initial_state = RNNInitialStateType.ZERO
        else:
            raise NotImplementedError

        generator = DoppelGANgerGenerator(
            feed_back=self._config["feed_back"],
            noise=self._config["noise"],
            attr_noise_type=self._config["attr_noise_type"],
            feature_noise_type=self._config["feature_noise_type"],
            feature_outputs=data_feature_outputs,
            attribute_outputs=data_attribute_outputs,
            real_attribute_mask=real_attribute_mask,
            sample_len=sample_len,
            feature_num_layers=self._config["gen_feature_num_layers"],
            feature_num_units=self._config["gen_feature_num_units"],
            attribute_num_layers=self._config["gen_attribute_num_layers"],
            attribute_num_units=self._config["gen_attribute_num_units"],
            rnn_mlp_num_layers=self._config["rnn_mlp_num_layers"],
            initial_state=initial_state,
            gt_lengths=gt_lengths,
            use_uniform_lengths=self._config["use_uniform_lengths"],
        )
        discriminator = Discriminator(
            scale=self._config["scale"],
            sn_mode=self._config["sn_mode"],
            num_layers=self._config["disc_num_layers"],
            num_units=self._config["disc_num_units"],
            leaky_relu=self._config["leaky_relu"],
        )
        if self._config["aux_disc"]:
            attr_discriminator = AttrDiscriminator(
                scale=self._config["scale"],
                sn_mode=self._config["sn_mode"],
                num_layers=self._config["attr_disc_num_layers"],
                num_units=self._config["attr_disc_num_units"],
                leaky_relu=self._config["leaky_relu"],
            )

        checkpoint_dir = os.path.join(self._config["result_folder"], "checkpoint")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        sample_dir = os.path.join(self._config["result_folder"], "sample")
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)
        time_path = os.path.join(self._config["result_folder"], "time.txt")

        if self._config["num_cores"] is None:
            run_config = tf.compat.v1.ConfigProto(device_count={"GPU": 0})
        else:
            num_cores = self._config["num_cores"]  # it means number of cores
            run_config = tf.compat.v1.ConfigProto(
                intra_op_parallelism_threads=num_cores,
                inter_op_parallelism_threads=num_cores,
                allow_soft_placement=True,
                device_count={"CPU": num_cores},
            )

        with tf.compat.v1.Session(config=run_config) as sess:
            gan = DoppelGANger(
                sess=sess,
                checkpoint_dir=checkpoint_dir,
                pretrain_dir=self._config.get("pretrain_dir", None),
                sample_dir=sample_dir,
                time_path=time_path,
                batch_size=self._config["batch_size"],
                iteration=self._config["iteration"],
                dataset=dataset,
                sample_len=sample_len,
                real_attribute_mask=real_attribute_mask,
                data_feature_outputs=data_feature_outputs,
                data_attribute_outputs=data_attribute_outputs,
                vis_freq=self._config["vis_freq"],
                vis_num_sample=self._config["vis_num_sample"],
                generator=generator,
                discriminator=discriminator,
                attr_discriminator=(
                    attr_discriminator if self._config["aux_disc"] else None
                ),
                d_gp_coe=self._config["d_gp_coe"],
                attr_d_gp_coe=(
                    self._config["attr_d_gp_coe"] if self._config["aux_disc"] else 0.0
                ),
                g_attr_d_coe=(
                    self._config["g_attr_d_coe"] if self._config["aux_disc"] else 0.0
                ),
                d_rounds=self._config["d_rounds"],
                g_rounds=self._config["g_rounds"],
                fix_feature_network=self._config["fix_feature_network"],
                g_lr=self._config["g_lr"],
                d_lr=self._config["d_lr"],
                attr_d_lr=(
                    self._config["attr_d_lr"] if self._config["aux_disc"] else 0.0
                ),
                extra_checkpoint_freq=self._config["extra_checkpoint_freq"],
                epoch_checkpoint_freq=self._config["epoch_checkpoint_freq"],
                num_packing=self._config["num_packing"],
                debug=self._config["debug"],
                combined_disc=self._config["combined_disc"],
                # DP-related
                dp_noise_multiplier=self._config["dp_noise_multiplier"],
                dp_l2_norm_clip=self._config["dp_l2_norm_clip"],
                # SN-related
                sn_mode=self._config["sn_mode"],
            )

            gan.build()
            gan.train(restore=self._config.get("restore", False))

        dataset.stop_data_loader()
        return True

    def _generate(
        self,
        input_train_data_folder,
        input_model_folder,
        output_syn_data_folder,
        log_folder,
    ):
        logger.debug(f"{self.__class__.__name__}.{inspect.stack()[0][3]}")

        self._config["result_folder"] = self._config.get(
            "result_folder", input_model_folder
        )
        self._config["dataset"] = self._config.get("dataset", input_train_data_folder)

        # If Ray is disabled, reset TF graph
        if not ray.config.enabled:
            tf.compat.v1.reset_default_graph()

        if self._config["given_data_attribute_flag"]:
            logger.debug("Generating from a given data attribute!")
            given_attr_npz_file = os.path.join(
                output_syn_data_folder,
                "attr_clean",
                "chunk_id-{}.npz".format(self._config["chunk_id"]),
            )

            if not os.path.exists(given_attr_npz_file):
                raise ValueError(f"Given data attribute file {given_attr_npz_file}")
            given_data_attribute = np.load(given_attr_npz_file)["data_attribute"]
            logger.debug(f"given_data_attribute: {given_data_attribute.shape}")
        else:
            logger.debug("Generating w/o given data attribute!")

        data_in_dir = self._config["dataset"]
        with open(os.path.join(data_in_dir, "data_feature_output.pkl"), "rb") as f:
            data_feature_outputs = pickle.load(f)
        with open(os.path.join(data_in_dir, "data_attribute_output.pkl"), "rb") as f:
            data_attribute_outputs = pickle.load(f)

        num_real_attribute = len(data_attribute_outputs)
        num_real_samples = get_config(
            "model.config.num_real_samples", default_value=None
        ) or len(
            [
                file
                for file in os.listdir(os.path.join(data_in_dir, "data_train_npz"))
                if file.endswith(".npz")
            ]
        )

        # add noise for DP generation
        if self._config["dp_noise_multiplier"] is not None:
            print("DP case: adding noise to # of flows")
            num_real_samples += int(estimate_flowlen_dp([num_real_samples])[0])

        # TODO: Why does it care about the type of the input data?
        dataset_type = get_config(
            "input_adapters.format_normalizer.dataset_type", default_value=None
        )
        if dataset_type == "netflow":
            self._config["generate_num_train_sample"] = int(1.25 * num_real_samples)
            self._config["generate_num_test_sample"] = 0
        elif dataset_type == "pcap":
            self._config["generate_num_train_sample"] = num_real_samples
            self._config["generate_num_test_sample"] = 0
        elif dataset_type == "zeeklog":
            self._config["generate_num_train_sample"] = int(1.25 * num_real_samples)
            self._config["generate_num_test_sample"] = 0
        else:
            self._config["generate_num_train_sample"] = self._config.get(
                "generate_num_train_sample", num_real_samples
            )
            self._config["generate_num_test_sample"] = 0

        if self._config["given_data_attribute_flag"]:
            self._config["generate_num_train_sample"] = np.shape(given_data_attribute)[
                0
            ]
            self._config["generate_num_test_sample"] = 0

        dataset = NetShareDataset(
            root=data_in_dir,
            config=self._config,
            data_attribute_outputs=data_attribute_outputs,
            data_feature_outputs=data_feature_outputs,
        )
        # Run dataset.sample_batch() once to intialize data_attribute_outputs
        # and data_feature_outputs
        logger.debug("Prepare sample batch")
        dataset.sample_batch(self._config["batch_size"])
        if (
            (dataset.data_attribute_outputs_train is None)
            or (dataset.data_feature_outputs_train is None)
            or (dataset.real_attribute_mask is None)
        ):
            logger.debug(
                f"dataset.data_attribute_outputs_train: {dataset.data_attribute_outputs_train}"
            )
            logger.debug(
                f"dataset.data_feature_outputs_train: {dataset.data_feature_outputs_train}"
            )
            logger.debug(f"dataset.real_attribute_mask: {dataset.real_attribute_mask}")
            raise Exception(
                "Dataset variables are not initialized properly for training purposes!"
            )
        logger.debug("finished preparing sample batch")

        sample_len = self._config["sample_len"]
        data_attribute_outputs = dataset.data_attribute_outputs_train
        data_feature_outputs = dataset.data_feature_outputs_train
        real_attribute_mask = dataset.real_attribute_mask
        gt_lengths = dataset.gt_lengths

        initial_state = None
        if self._config["initial_state"] == "variable":
            initial_state = RNNInitialStateType.VARIABLE
        elif self._config["initial_state"] == "random":
            initial_state = RNNInitialStateType.RANDOM
        elif self._config["initial_state"] == "zero":
            initial_state = RNNInitialStateType.ZERO
        else:
            raise NotImplementedError

        generator = DoppelGANgerGenerator(
            feed_back=self._config["feed_back"],
            noise=self._config["noise"],
            attr_noise_type=self._config["attr_noise_type"],
            feature_noise_type=self._config["feature_noise_type"],
            feature_outputs=data_feature_outputs,
            attribute_outputs=data_attribute_outputs,
            real_attribute_mask=real_attribute_mask,
            sample_len=sample_len,
            feature_num_layers=self._config["gen_feature_num_layers"],
            feature_num_units=self._config["gen_feature_num_units"],
            attribute_num_layers=self._config["gen_attribute_num_layers"],
            attribute_num_units=self._config["gen_attribute_num_units"],
            rnn_mlp_num_layers=self._config["rnn_mlp_num_layers"],
            initial_state=initial_state,
            gt_lengths=gt_lengths,
            use_uniform_lengths=self._config["use_uniform_lengths"],
        )
        discriminator = Discriminator(
            scale=self._config["scale"],
            sn_mode=self._config["sn_mode"],
            num_layers=self._config["disc_num_layers"],
            num_units=self._config["disc_num_units"],
            leaky_relu=self._config["leaky_relu"],
        )
        if self._config["aux_disc"]:
            attr_discriminator = AttrDiscriminator(
                scale=self._config["scale"],
                sn_mode=self._config["sn_mode"],
                num_layers=self._config["attr_disc_num_layers"],
                num_units=self._config["attr_disc_num_units"],
                leaky_relu=self._config["leaky_relu"],
            )

        checkpoint_dir = os.path.join(input_model_folder, "checkpoint")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        sample_dir = os.path.join(input_model_folder, "sample")
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)

        time_path = os.path.join(input_model_folder, "time.txt")

        if self._config["num_cores"] is None:
            run_config = tf.compat.v1.ConfigProto()
        else:
            num_cores = self._config["num_cores"]  # it means number of cores
            run_config = tf.compat.v1.ConfigProto(
                intra_op_parallelism_threads=num_cores,
                inter_op_parallelism_threads=num_cores,
                allow_soft_placement=True,
                device_count={"CPU": num_cores},
            )

        with tf.compat.v1.Session(config=run_config) as sess:
            gan = DoppelGANger(
                sess=sess,
                checkpoint_dir=checkpoint_dir,
                pretrain_dir=self._config.get("pretrain_dir", None),
                sample_dir=sample_dir,
                time_path=time_path,
                batch_size=self._config["batch_size"],
                iteration=self._config["iteration"],
                dataset=dataset,
                sample_len=sample_len,
                real_attribute_mask=real_attribute_mask,
                data_feature_outputs=data_feature_outputs,
                data_attribute_outputs=data_attribute_outputs,
                vis_freq=self._config["vis_freq"],
                vis_num_sample=self._config["vis_num_sample"],
                generator=generator,
                discriminator=discriminator,
                attr_discriminator=(
                    attr_discriminator if self._config["aux_disc"] else None
                ),
                d_gp_coe=self._config["d_gp_coe"],
                attr_d_gp_coe=(
                    self._config["attr_d_gp_coe"] if self._config["aux_disc"] else 0.0
                ),
                g_attr_d_coe=(
                    self._config["g_attr_d_coe"] if self._config["aux_disc"] else 0.0
                ),
                d_rounds=self._config["d_rounds"],
                g_rounds=self._config["g_rounds"],
                fix_feature_network=self._config["fix_feature_network"],
                g_lr=self._config["g_lr"],
                d_lr=self._config["d_lr"],
                attr_d_lr=(
                    self._config["attr_d_lr"] if self._config["aux_disc"] else 0.0
                ),
                extra_checkpoint_freq=self._config["extra_checkpoint_freq"],
                epoch_checkpoint_freq=self._config["epoch_checkpoint_freq"],
                num_packing=self._config["num_packing"],
                debug=self._config["debug"],
                combined_disc=self._config["combined_disc"],
                # DP-related
                dp_noise_multiplier=self._config["dp_noise_multiplier"],
                dp_l2_norm_clip=self._config["dp_l2_norm_clip"],
                # SN-related
                sn_mode=self._config["sn_mode"],
            )

            gan.build()

            total_generate_num_sample = (
                self._config["generate_num_train_sample"]
                + self._config["generate_num_test_sample"]
            )
            logger.debug(f"total generated sample: {total_generate_num_sample}")
            if total_generate_num_sample == 0:
                raise ValueError(
                    f"Generating given data attribute: {self._config['given_data_attribute_flag']}. # of total generated samples is ZERO."
                )

            (
                batch_data_attribute,
                batch_data_feature,
                batch_data_gen_flag,
            ) = dataset.sample_batch(self._config["batch_size"])
            if batch_data_feature.shape[1] % sample_len != 0:
                raise Exception("length must be a multiple of sample_len")
            length = int(batch_data_feature.shape[1] / sample_len)
            real_attribute_input_noise = gan.gen_attribute_input_noise(
                total_generate_num_sample
            )
            addi_attribute_input_noise = gan.gen_attribute_input_noise(
                total_generate_num_sample
            )
            feature_input_noise = gan.gen_feature_input_noise(
                total_generate_num_sample, length
            )
            input_data = gan.gen_feature_input_data_free(total_generate_num_sample)

            last_iteration_found = False
            iteration_range = list(
                range(
                    self._config["extra_checkpoint_freq"] - 1,
                    self._config["iteration"],
                    self._config["extra_checkpoint_freq"],
                )
            )
            # reverse list in place
            iteration_range.reverse()

            generatedSamples_per_epoch = 1

            for iteration_id in iteration_range:
                if last_iteration_found and (
                    (not self._config["given_data_attribute_flag"])
                    and (not self._config.get("single_chunk_flag", False))
                ):
                    break

                logger.debug("Processing iteration_id: {}".format(iteration_id))
                mid_checkpoint_dir = os.path.join(
                    checkpoint_dir, "iteration_id-{}".format(iteration_id)
                )
                if not os.path.exists(mid_checkpoint_dir):
                    logger.debug(
                        "Middle checkpoint dir not found {}".format(mid_checkpoint_dir)
                    )
                    continue
                else:
                    last_iteration_found = True
                for generated_samples_idx in range(generatedSamples_per_epoch):
                    logger.debug(
                        "generate {}-th sample from iteration_id-{}".format(
                            generated_samples_idx + 1, iteration_id
                        )
                    )

                    gan.load(mid_checkpoint_dir)

                    # specify given_attribute parameter, if you want to generate
                    # data according to an attribute
                    if self._config["given_data_attribute_flag"]:
                        features, attributes, gen_flags, lengths = gan.sample_from(
                            real_attribute_input_noise,
                            addi_attribute_input_noise,
                            feature_input_noise,
                            input_data,
                            given_attribute=given_data_attribute,
                        )
                    else:
                        features, attributes, gen_flags, lengths = gan.sample_from(
                            real_attribute_input_noise,
                            addi_attribute_input_noise,
                            feature_input_noise,
                            input_data,
                        )

                    split = self._config["generate_num_train_sample"]

                    if self._config["self_norm"]:
                        features, attributes = renormalize_per_sample(
                            features,
                            attributes,
                            data_feature_outputs,
                            data_attribute_outputs,
                            gen_flags,
                            num_real_attribute=num_real_attribute,
                        )

                    # if self._config.get("save_without_chunk", False):
                    #     save_path = os.path.join(
                    #         output_syn_data_folder, f"iteration_id-{iteration_id}"
                    #     )
                    #     os.makedirs(save_path, exist_ok=True)
                    #     np.savez(
                    #         os.path.join(save_path, "data.npz"),
                    #         data_attribute=attributes,
                    #         data_feature=features,
                    #         data_gen_flag=gen_flags,
                    #     )

                    if not self._config["given_data_attribute_flag"]:
                        # multi-chunk: generate attributes only
                        if not self._config.get("single_chunk_flag", False):
                            save_path = os.path.join(output_syn_data_folder, "attr_raw")
                            os.makedirs(save_path, exist_ok=True)
                            np.savez(
                                os.path.join(
                                    save_path,
                                    "chunk_id-{}.npz".format(self._config["chunk_id"]),
                                ),
                                data_attribute=attributes[0:split],
                            )
                    # multi-chunk: generate features given attribute
                    # single-chunk: co-generate attributes and features
                    if self._config["given_data_attribute_flag"] or (
                        (not self._config["given_data_attribute_flag"])
                        and self._config.get("single_chunk_flag", False)
                    ):
                        logger.info(f"Generate to {output_syn_data_folder}")
                        save_path = os.path.join(
                            output_syn_data_folder,
                            "feat_raw",
                            f"chunk_id-{self._config['chunk_id']}",
                        )
                        os.makedirs(save_path, exist_ok=True)
                        np.savez(
                            os.path.join(
                                save_path,
                                f"iteration_id-{iteration_id}.npz",
                            ),
                            data_attribute=attributes,
                            data_feature=features,
                            data_gen_flag=gen_flags,
                            config=self._config,
                        )

        dataset.stop_data_loader()

        return True
