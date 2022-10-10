from abc import ABC, abstractmethod
import os

from netshare.utils import Tee


class ModelManager(ABC):
    def __init__(self, config):
        self._config = config

    @abstractmethod
    def _train(self, input_train_data_folder, output_model_folder,
               log_folder, create_new_model, model_config):
        ...

    @abstractmethod
    def _generate(self,
                  input_train_data_folder, input_model_folder,
                  output_syn_data_folder, log_folder,
                  create_new_model, model_config):
        ...

    def train(self, input_train_data_folder, output_model_folder, log_folder,
              create_new_model, model_config):
        stdout_log_path = os.path.join(log_folder, 'train.stdout.log')
        stderr_log_path = os.path.join(log_folder, 'train.stderr.log')
        with Tee(stdout_path=stdout_log_path, stderr_path=stderr_log_path):
            return self._train(
                input_train_data_folder=input_train_data_folder,
                output_model_folder=output_model_folder,
                log_folder=log_folder,
                create_new_model=create_new_model,
                model_config=model_config)

    def generate(self,
                 input_train_data_folder, input_model_folder,
                 output_syn_data_folder, log_folder,
                 create_new_model, model_config):
        stdout_log_path = os.path.join(log_folder, 'generate.stdout.log')
        stderr_log_path = os.path.join(log_folder, 'generate.stderr.log')
        with Tee(stdout_path=stdout_log_path, stderr_path=stderr_log_path):
            return self._generate(
                input_train_data_folder=input_train_data_folder,
                input_model_folder=input_model_folder,
                output_syn_data_folder=output_syn_data_folder,
                log_folder=log_folder,
                create_new_model=create_new_model,
                model_config=model_config)
