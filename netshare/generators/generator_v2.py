import os
import copy
from netshare.logger import logger
import shutil

from config_io import Config

import netshare.model_managers as model_managers
import netshare.models as models
import netshare.dashboard as dashboard
from netshare.dashboard.dist_metrics import (
    run_netflow_qualitative_plots_dashboard,
    run_pcap_qualitative_plots_dashboard,
)
from netshare.configs import default as default_configs
from netshare.post_process.post_process import post_process
from netshare.pre_process.pre_process import pre_process


class GeneratorV2(object):
    def __init__(self, config):
        self._config = Config.load_from_file(
            config, default_search_paths=default_configs.__path__
        )
        config = copy.deepcopy(self._config)

        global_config = self._config["global_config"]

        self._overwrite = global_config["overwrite"]

        model_manager_class = getattr(model_managers, config["model_manager"]["class"])
        model_manager_config = Config(global_config)
        model_manager_config.update(config["model_manager"]["config"])
        self._model_manager = model_manager_class(config=model_manager_config)

        model_class = getattr(models, config["model"]["class"])
        model_config = config["model"]["config"]
        self._model = model_class
        self._model_config = model_config

        dashboard_class_path = os.path.dirname(dashboard.__file__)
        self.static_folder_for_vis = os.path.join(dashboard_class_path, "static")
        self.figure_stored_relative_folder_for_vis = "tmp"
        self.static_figure_folder_for_vis = os.path.join(
            self.static_folder_for_vis, self.figure_stored_relative_folder_for_vis
        )

    def _get_pre_processed_data_folder(self, work_folder):
        return os.path.join(work_folder, "pre_processed_data")

    def _get_post_processed_data_folder(self, work_folder):
        return os.path.join(work_folder, "post_processed_data")

    def _get_generated_data_folder(self, work_folder):
        return os.path.join(work_folder, "generated_data")

    def _get_model_folder(self, work_folder):
        return os.path.join(work_folder, "models")

    def _get_visualization_folder(self, work_folder):
        return os.path.join(work_folder, "visulization")

    def _get_pre_processed_data_log_folder(self, work_folder):
        return os.path.join(work_folder, "logs", "pre_processed_data")

    def _get_post_processed_data_log_folder(self, work_folder):
        return os.path.join(work_folder, "logs", "post_processed_data")

    def _get_generated_data_log_folder(self, work_folder):
        return os.path.join(work_folder, "logs", "generated_data")

    def _get_model_log_folder(self, work_folder):
        return os.path.join(work_folder, "logs", "models")

    def _post_process(self):
        return post_process(config=self._config)

    def _train(self, input_train_data_folder, output_model_folder, log_folder):
        if not self._check_folder(output_model_folder):
            return False
        if not self._check_folder(log_folder):
            return False
        return self._model_manager.train(
            input_train_data_folder=input_train_data_folder,
            output_model_folder=output_model_folder,
            log_folder=log_folder,
            create_new_model=self._model,
            model_config=self._model_config,
        )

    def _generate(
        self,
        input_train_data_folder,
        input_model_folder,
        output_syn_data_folder,
        log_folder,
    ):
        if not self._check_folder(output_syn_data_folder):
            return False
        if not self._check_folder(log_folder):
            return False
        return self._model_manager.generate(
            input_train_data_folder=input_train_data_folder,
            input_model_folder=input_model_folder,
            output_syn_data_folder=output_syn_data_folder,
            log_folder=log_folder,
            create_new_model=self._model,
            model_config=self._model_config,
        )

    def _check_folder(self, folder):
        if os.path.exists(folder):
            if self._overwrite:
                logger.warning(
                    f"{folder} already exists. You are overwriting the results."
                )
                return True
            else:
                logger.info(
                    f"{folder} already exists. To avoid overwriting the "
                    "results, please change the work_folder"
                )
                return False
            return False
        os.makedirs(folder)
        return True

    def _copy_figures_to_dashboard_static_folder(self, folder):

        if os.path.exists(self.static_figure_folder_for_vis):
            shutil.rmtree(self.static_figure_folder_for_vis)
        shutil.copytree(folder, self.static_figure_folder_for_vis)

    def generate(self, work_folder):
        work_folder = os.path.expanduser(work_folder)
        if not self._generate(
            input_train_data_folder=self._get_pre_processed_data_folder(work_folder),
            input_model_folder=self._get_model_folder(work_folder),
            output_syn_data_folder=self._get_generated_data_folder(work_folder),
            log_folder=self._get_generated_data_log_folder(work_folder),
        ):
            print("Failed to generate synthetic data")
            return False
        self._post_process()
        return True

    def train(self, work_folder):
        self._check_folder(self._get_pre_processed_data_folder(work_folder))
        pre_process(
            config=self._config,
            target_dir=self._get_pre_processed_data_folder(work_folder),
        )
        if not self._train(
            input_train_data_folder=self._get_pre_processed_data_folder(work_folder),
            output_model_folder=self._get_model_folder(work_folder),
            log_folder=self._get_model_log_folder(work_folder),
        ):
            print("Failed to train the model")
            return False
        return True

    def train_and_generate(self, work_folder):
        work_folder = os.path.expanduser(work_folder)
        self.train(work_folder)
        return self.generate(work_folder)

    def visualize(self, work_folder):
        work_folder = os.path.expanduser(work_folder)
        os.makedirs(self._get_visualization_folder(work_folder), exist_ok=True)

        original_data_file = self._config["global_config"]["original_data_file"]
        dataset_type = self._config["global_config"]["dataset_type"]
        original_data_path = os.path.dirname(original_data_file)

        # Check if pre-generated synthetic data exists
        if os.path.exists(os.path.join(original_data_path, "syn.csv")):
            print("Pre-generated synthetic data exists!")
            syn_data_path = os.path.join(original_data_path, "syn.csv")
        # Check if self-generated synthetic data exists
        elif os.path.exists(
            os.path.join(self._get_post_processed_data_folder(work_folder), "syn.csv")
        ):
            print("Self-generated synthetic data exists!")
            syn_data_path = os.path.join(
                self._get_post_processed_data_folder(work_folder), "syn.csv"
            )
        else:
            raise ValueError(
                "Neither pre-generated OR self-generated synthetic data exists!"
            )

        if dataset_type == "netflow":
            run_netflow_qualitative_plots_dashboard(
                raw_data_path=original_data_file,
                syn_data_path=syn_data_path,
                plot_dir=self._get_visualization_folder(work_folder),
            )
        elif dataset_type == "pcap":
            run_pcap_qualitative_plots_dashboard(
                raw_data_path=os.path.join(
                    self._get_pre_processed_data_folder(work_folder), "raw.csv"
                ),
                syn_data_path=syn_data_path,
                plot_dir=self._get_visualization_folder(work_folder),
            )

        self._copy_figures_to_dashboard_static_folder(
            self._get_visualization_folder(work_folder)
        )
        dashboard_class = getattr(dashboard, "Dashboard")
        dashboard_class(
            self._get_visualization_folder(work_folder),
            self.figure_stored_relative_folder_for_vis,
        )

        return True
