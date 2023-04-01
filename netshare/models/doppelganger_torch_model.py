import sys
import os
import json
import inspect
import numpy as np

from .model import Model
from netshare.utils import output
# from gan import output  # NOQA
# sys.modules["output"] = output  # NOQA
from .doppelganger_torch.doppelganger import DoppelGANger  # NOQA
from .doppelganger_torch.util import add_gen_flag, normalize_per_sample, renormalize_per_sample  # NOQA
from .doppelganger_torch.load_data import load_data  # NOQA


class DoppelGANgerTorchModel(Model):
    def _train(self, input_train_data_folder, output_model_folder, log_folder):
        print(f"{self.__class__.__name__}.{inspect.stack()[0][3]}")

        self._config["result_folder"] = getattr(
            self._config, "result_folder", output_model_folder)
        self._config["dataset"] = getattr(
            self._config, "dataset", input_train_data_folder)

        print("Currently training with config:", self._config)
        # save config to the result folder
        with open(os.path.join(
                self._config["result_folder"],
                "config.json"), 'w') as fout:
            json.dump(self._config, fout)

        # load data
        (
            data_feature,
            data_attribute,
            data_gen_flag,
            data_feature_outputs,
            data_attribute_outputs,
        ) = load_data(
            path=self._config["dataset"],
            sample_len=self._config["sample_len"])
        num_real_attribute = len(data_attribute_outputs)

        # self-norm if applicable
        if self._config["self_norm"]:
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
            data_feature, data_gen_flag, data_feature_outputs, self._config["sample_len"]
        )

        # create directories
        checkpoint_dir = os.path.join(
            self._config["result_folder"],
            "checkpoint")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        sample_dir = os.path.join(self._config["result_folder"], "sample")
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)
        time_path = os.path.join(self._config["result_folder"], "time.txt")

        dg = DoppelGANger(
            checkpoint_dir=checkpoint_dir,
            sample_dir=self._config["sample_len"],
            time_path=time_path,
            batch_size=self._config["batch_size"],
            real_attribute_mask=real_attribute_mask,
            max_sequence_len=data_feature.shape[1],
            sample_len=self._config["sample_len"],
            data_feature_outputs=data_feature_outputs,
            data_attribute_outputs=data_attribute_outputs,
            vis_freq=self._config["vis_freq"],
            vis_num_sample=self._config["vis_num_sample"],
            d_rounds=self._config["d_rounds"],
            g_rounds=self._config["g_rounds"],
            d_gp_coe=self._config["d_gp_coe"],
            attr_d_gp_coe=self._config["attr_d_gp_coe"],
            g_attr_d_coe=self._config["g_attr_d_coe"],
            use_adaptive_rolling=self._config["use_adaptive_rolling"],
            epoch_checkpoint_freq=self._config["epoch_checkpoint_freq"],
            restore=getattr(self._config, "restore", False),
            pretrain_dir=self._config["pretrain_dir"]
        )

        dg.train(
            epochs=self._config["epochs"],
            data_feature=data_feature,
            data_attribute=data_attribute,
            data_gen_flag=data_gen_flag,
        )

    def _generate(self, input_train_data_folder,
                  input_model_folder, output_syn_data_folder, log_folder):
        print(f"{self.__class__.__name__}.{inspect.stack()[0][3]}")

        self._config["result_folder"] = getattr(
            self._config, "result_folder", input_model_folder)
        self._config["dataset"] = getattr(
            self._config, "dataset", input_train_data_folder)

        print("Currently generating with config:", self._config)

        # load data
        (
            data_feature,
            data_attribute,
            data_gen_flag,
            data_feature_outputs,
            data_attribute_outputs,
        ) = load_data(
            path=self._config["dataset"],
            sample_len=self._config["sample_len"])
        num_real_attribute = len(data_attribute_outputs)

        # self-norm if applicable
        if self._config["self_norm"]:
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
            data_feature, data_gen_flag, data_feature_outputs, self._config["sample_len"]
        )

        # create directories
        checkpoint_dir = os.path.join(
            self._config["result_folder"],
            "checkpoint")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        sample_dir = os.path.join(self._config["result_folder"], "sample")
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)
        time_path = os.path.join(self._config["result_folder"], "time.txt")

        dg = DoppelGANger(
            checkpoint_dir=checkpoint_dir,
            sample_dir=self._config["sample_len"],
            time_path=time_path,
            batch_size=self._config["batch_size"],
            real_attribute_mask=real_attribute_mask,
            max_sequence_len=data_feature.shape[1],
            sample_len=self._config["sample_len"],
            data_feature_outputs=data_feature_outputs,
            data_attribute_outputs=data_attribute_outputs,
            vis_freq=self._config["vis_freq"],
            vis_num_sample=self._config["vis_num_sample"],
            d_rounds=self._config["d_rounds"],
            g_rounds=self._config["g_rounds"],
            d_gp_coe=self._config["d_gp_coe"],
            attr_d_gp_coe=self._config["attr_d_gp_coe"],
            g_attr_d_coe=self._config["g_attr_d_coe"],
            use_adaptive_rolling=self._config["use_adaptive_rolling"],
            epoch_checkpoint_freq=self._config["epoch_checkpoint_freq"],
            restore=getattr(self._config, "restore", False),
            pretrain_dir=self._config["pretrain_dir"]
        )

        if self._config["given_data_attribute_flag"]:
            print("Generating from a given data attribute!")
            given_attr_npz_file = os.path.join(
                output_syn_data_folder,
                "attr_clean",
                "chunk_id-{}.npz".format(self._config["chunk_id"]))

            if not os.path.exists(given_attr_npz_file):
                raise ValueError(
                    f"Given data attribute file {given_attr_npz_file}")
            given_data_attribute = np.load(given_attr_npz_file)[
                "data_attribute"]
            given_data_attribute_discrete = np.load(given_attr_npz_file)[
                "data_attribute_discrete"]
            print("given_data_attribute:", given_data_attribute.shape)
            print("given_data_attribute_discrete:",
                  given_data_attribute_discrete)
        else:
            print("Generating w/o given data attribute!")
            given_data_attribute = None
            given_data_attribute_discrete = None

        last_iteration_found = False
        epoch_range = list(
            range(
                self._config["epoch_checkpoint_freq"] - 1,
                self._config["epochs"],
                self._config["epoch_checkpoint_freq"],
            )
        )
        # reverse list in place
        epoch_range.reverse()
        generatedSamples_per_epoch = 1

        for epoch_id in epoch_range:
            if last_iteration_found == True:
                break

            print("Processing epoch_id: {}".format(epoch_id))
            mid_checkpoint_dir = os.path.join(
                checkpoint_dir, "epoch_id-{}.pt".format(epoch_id)
            )
            if not os.path.exists(mid_checkpoint_dir):
                print("Not found {}".format(mid_checkpoint_dir))
                continue
            else:
                last_iteration_found = True
            for generated_samples_idx in range(generatedSamples_per_epoch):
                print(
                    "generate {}-th sample from epoch_id-{}".format(
                        generated_samples_idx + 1, epoch_id
                    )
                )

                dg.load(mid_checkpoint_dir)
                print("Finished loading")

                (
                    features, attributes, attributes_discrete, gen_flags,
                    lengths) = dg.generate(
                    num_samples=data_attribute.shape[0]
                    if (given_data_attribute is None)
                    and (given_data_attribute_discrete is None) else
                    given_data_attribute.shape[0],
                    given_attribute=given_data_attribute,
                    given_attribute_discrete=given_data_attribute_discrete)

                print(features.shape)
                print(attributes.shape)
                print(gen_flags.shape)
                print(lengths.shape)

                if self._config["self_norm"]:
                    features, attributes = renormalize_per_sample(
                        features,
                        attributes,
                        data_feature_outputs,
                        data_attribute_outputs,
                        gen_flags,
                        num_real_attribute=num_real_attribute,
                    )

                    print(features.shape)
                    print(attributes.shape)

                if getattr(self._config, "save_without_chunk", False):
                    save_path = os.path.join(
                        output_syn_data_folder,
                        f"epoch_id-{epoch_id}")
                    os.makedirs(save_path, exist_ok=True)
                    np.savez(
                        os.path.join(
                            save_path,
                            "data.npz"),
                        data_attribute=attributes,
                        data_feature=features,
                        data_gen_flag=gen_flags)
                elif not self._config["given_data_attribute_flag"]:
                    save_path = os.path.join(
                        output_syn_data_folder, "attr_raw")
                    os.makedirs(save_path, exist_ok=True)
                    np.savez(
                        os.path.join(
                            save_path,
                            "chunk_id-{}.npz".format(
                                self._config["chunk_id"])
                        ),
                        data_attribute=attributes,
                        data_attribute_discrete=attributes_discrete
                    )
                    print(os.path.join(
                        save_path,
                        "chunk_id-{}.npz".format(
                            self._config["chunk_id"])
                    ))
                else:
                    save_path = os.path.join(
                        output_syn_data_folder, "feat_raw")
                    os.makedirs(save_path, exist_ok=True)
                    np.savez(
                        os.path.join(
                            save_path,
                            f"chunk_id-{self._config['chunk_id']}",
                            f"epoch_id-{epoch_id}.npz"
                        ),
                        data_attribute=attributes,
                        data_feature=features,
                        data_gen_flag=gen_flags,
                        config=self._config
                    )
