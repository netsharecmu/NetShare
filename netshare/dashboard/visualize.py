import os
import shutil

from netshare.configs import get_config
from netshare.dashboard import static
from netshare.dashboard.dashboard import Dashboard
from netshare.dashboard.dist_metrics import (
    run_netflow_qualitative_plots_dashboard,
    run_pcap_qualitative_plots_dashboard,
)
from netshare.logger import logger

VISUALIZE_DIR = "tmp"


def _copy_figures_to_dashboard_static_folder(from_dir, to_dir):
    if os.path.exists(to_dir):
        shutil.rmtree(to_dir)
    shutil.copytree(from_dir, to_dir)


def visualize(generated_data_dir: str, target_dir: str, original_data_dir: str) -> None:
    os.makedirs(target_dir, exist_ok=True)
    dashboard_class_path = os.path.dirname(static.__file__)
    static_figure_folder_for_vis = os.path.join(dashboard_class_path, VISUALIZE_DIR)

    original_data_file = get_config("global_config.original_data_file")
    dataset_type = get_config("global_config.dataset_type")
    original_data_path = os.path.dirname(original_data_file)

    # Check if pre-generated synthetic data exists
    if os.path.exists(os.path.join(original_data_path, "syn.csv")):
        logger.debug("Pre-generated synthetic data exists!")
        syn_data_path = os.path.join(original_data_path, "syn.csv")
    # Check if self-generated synthetic data exists
    elif os.path.exists(os.path.join(generated_data_dir, "syn.csv")):
        logger.debug("Self-generated synthetic data exists!")
        syn_data_path = os.path.join(generated_data_dir, "syn.csv")
    else:
        raise ValueError(
            "Neither pre-generated OR self-generated synthetic data exists!"
        )

    if dataset_type == "netflow":
        run_netflow_qualitative_plots_dashboard(
            raw_data_path=original_data_file,
            syn_data_path=syn_data_path,
            plot_dir=target_dir,
        )
    elif dataset_type == "pcap":
        run_pcap_qualitative_plots_dashboard(
            raw_data_path=os.path.join(original_data_dir, "raw.csv"),
            syn_data_path=syn_data_path,
            plot_dir=target_dir,
        )

    # _copy_figures_to_dashboard_static_folder
    if os.path.exists(static_figure_folder_for_vis):
        shutil.rmtree(static_figure_folder_for_vis)
    shutil.copytree(target_dir, static_figure_folder_for_vis)

    Dashboard(target_dir, VISUALIZE_DIR)
