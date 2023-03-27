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
        )

        dg.train(
            epochs=10,
            data_feature=data_feature,
            data_attribute=data_attribute,
            data_gen_flag=data_gen_flag,
        )

    def _generate(self, input_train_data_folder,
                  input_model_folder, output_syn_data_folder, log_folder):
        print(f"{self.__class__.__name__}.{inspect.stack()[0][3]}")
