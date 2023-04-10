import os
import re
import copy
import warnings
import pandas as pd

import netshare.pre_post_processors as pre_post_processors
import netshare.model_managers as model_managers
import netshare.models as models
from ..pre_post_processors.netshare.util import create_sdmetrics_config

from config_io import Config
from sdmetrics.reports.timeseries import QualityReport
from ..configs import default as default_configs


class Generator(object):
    def __init__(self, config):
        self._config = Config.load_from_file(
            config,
            default_search_paths=default_configs.__path__)
        config = copy.deepcopy(self._config)

        global_config = self._config["global_config"]

        if 'original_data_folder' in global_config and \
                'file_extension' not in global_config:
            raise ValueError('Input is a folder. '
                             'Intended file extensions must be specified with '
                             '`file_extension=<.ext>` (e.g., `.pcap`, `.csv`')

        if 'original_data_folder' in global_config and \
                "original_data_file" in global_config:
            raise ValueError(
                'Input can be either a single file or \
                    a folder (with multiple valid files)!')
        self._ori_data_path = global_config['original_data_folder'] \
            if 'original_data_folder' in global_config \
            else global_config['original_data_file']
        self._overwrite = global_config['overwrite']

        pre_post_processor_class = getattr(
            pre_post_processors, config['pre_post_processor']['class'])
        pre_post_processor_config = Config(global_config)
        pre_post_processor_config.update(
            config['pre_post_processor']['config'])
        self._pre_post_processor = pre_post_processor_class(
            config=pre_post_processor_config)

        model_manager_class = getattr(
            model_managers, config['model_manager']['class'])
        model_manager_config = Config(global_config)
        model_manager_config.update(config['model_manager']['config'])
        self._model_manager = model_manager_class(config=model_manager_config)

        model_class = getattr(models, config['model']['class'])
        model_config = config['model']['config']
        self._model = model_class
        self._model_config = model_config

    def _get_pre_processed_data_folder(self, work_folder):
        return os.path.join(work_folder, 'pre_processed_data')

    def _get_post_processed_data_folder(self, work_folder):
        return os.path.join(work_folder, 'post_processed_data')

    def _get_generated_data_folder(self, work_folder):
        return os.path.join(work_folder, 'generated_data')

    def _get_model_folder(self, work_folder):
        return os.path.join(work_folder, 'models')

    def _get_visualization_folder(self, work_folder):
        return os.path.join(work_folder, "visulization")

    def _get_pre_processed_data_log_folder(self, work_folder):
        return os.path.join(work_folder, 'logs', 'pre_processed_data')

    def _get_post_processed_data_log_folder(self, work_folder):
        return os.path.join(work_folder, 'logs', 'post_processed_data')

    def _get_generated_data_log_folder(self, work_folder):
        return os.path.join(work_folder, 'logs', 'generated_data')

    def _get_model_log_folder(self, work_folder):
        return os.path.join(work_folder, 'logs', 'models')

    def _pre_process(self, input_folder, output_folder, log_folder):
        if not self._check_folder(output_folder):
            return False
        if not self._check_folder(log_folder):
            return False
        return self._pre_post_processor.pre_process(
            input_folder=input_folder,
            output_folder=output_folder,
            log_folder=log_folder)

    def _post_process(self, input_folder, output_folder,
                      pre_processed_data_folder, log_folder):
        if not self._check_folder(output_folder):
            return False
        if not self._check_folder(log_folder):
            return False
        return self._pre_post_processor.post_process(
            input_folder=input_folder,
            output_folder=output_folder,
            pre_processed_data_folder=pre_processed_data_folder,
            log_folder=log_folder)

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
            model_config=self._model_config)

    def _generate(self, input_train_data_folder,
                  input_model_folder, output_syn_data_folder, log_folder):
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
            model_config=self._model_config)

    def _check_folder(self, folder):
        if os.path.exists(folder):
            if self._overwrite:
                warnings.warn(
                    f'{folder} already exists. '
                    'You are overwriting the results.')
                return True
            else:
                print(
                    f'{folder} already exists. To avoid overwriting the '
                    'results, please change the work_folder')
                return False
            return False
        os.makedirs(folder)
        return True

    def generate(self, work_folder):
        work_folder = os.path.expanduser(work_folder)
        if not self._generate(
                input_train_data_folder=self._get_pre_processed_data_folder(
                    work_folder),
                input_model_folder=self._get_model_folder(work_folder),
                output_syn_data_folder=self._get_generated_data_folder(
                    work_folder),
                log_folder=self._get_generated_data_log_folder(work_folder)):
            print('Failed to generate synthetic data')
            return False
        if not self._post_process(
                input_folder=self._get_generated_data_folder(work_folder),
                output_folder=self._get_post_processed_data_folder(
                    work_folder),
                log_folder=self._get_post_processed_data_log_folder(
                    work_folder),
                pre_processed_data_folder=self._get_pre_processed_data_folder(
                    work_folder)):
            print('Failed to post-process data')
            return False
        print(f'Generated data is at '
              f'{self._get_post_processed_data_folder(work_folder)}')
        return True

    def train(self, work_folder):
        work_folder = os.path.expanduser(work_folder)
        if not self._pre_process(
                input_folder=self._ori_data_path,
                output_folder=self._get_pre_processed_data_folder(work_folder),
                log_folder=self._get_pre_processed_data_log_folder(
                    work_folder)):
            print('Failed to pre-process data')
            return False
        if not self._train(
                input_train_data_folder=self._get_pre_processed_data_folder(
                    work_folder),
                output_model_folder=self._get_model_folder(work_folder),
                log_folder=self._get_model_log_folder(work_folder)):
            print('Failed to train the model')
            return False
        return True

    def train_and_generate(self, work_folder):
        work_folder = os.path.expanduser(work_folder)
        if not self.train(work_folder):
            return False
        if not self.generate(work_folder):
            return False
        return True

    def visualize(self, work_folder):
        work_folder = os.path.expanduser(work_folder)
        os.makedirs(self._get_visualization_folder(work_folder), exist_ok=True)
        real_data = pd.read_csv(
            os.path.join(
                self._get_pre_processed_data_folder(work_folder), "raw.csv"))
        # Find synthetic data with the largest ID
        syn_data_list = [
            f
            for f in os.listdir(
                self._get_post_processed_data_folder(work_folder))
            if f.endswith('.csv')]
        id_pattern = re.compile(r'id-(\d+).csv')
        ids = [int(id_pattern.search(filename).group(1))
               for filename in syn_data_list if id_pattern.search(filename)]
        # Find the largest ID
        largest_id = max(ids)
        # Find the filename corresponding to the largest ID
        filename_with_largest_id = [
            filename for filename in syn_data_list
            if f'id-{largest_id}.csv' in filename][0]
        print(
            f'The filename with the largest ID is: {filename_with_largest_id}')
        synthetic_data = pd.read_csv(os.path.join(
            self._get_post_processed_data_folder(work_folder),
            filename_with_largest_id
        ))

        # Visualize the real data and synthetic data
        pre_post_processor_config = Config(self._config["global_config"])
        pre_post_processor_config.update(
            self._config['pre_post_processor']['config'])
        sdmetrics_config = create_sdmetrics_config(
            pre_post_processor_config,
            comparison_type='both'
        )
        my_report = QualityReport(
            config_dict=sdmetrics_config['config'])
        my_report.generate(real_data[synthetic_data.columns], synthetic_data,
                           sdmetrics_config['metadata'])
        my_report.visualize()
