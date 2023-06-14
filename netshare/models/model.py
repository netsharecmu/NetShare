from abc import ABC, abstractmethod
import os

from netshare.utils import Tee


class Model(ABC):
    def __init__(self, config):
        self._config = config

    @abstractmethod
    def _train(self, input_train_data_folder, output_model_folder, log_folder):
        ...

    @abstractmethod
    def _train_distributed(self, input_train_data_folder, output_model_folder, log_folder):
        ...

    @abstractmethod
    def _generate(self, input_train_data_folder,
                  input_model_folder, output_syn_data_folder, log_folder):
        ...

    def train(self, input_train_data_folder, output_model_folder, log_folder, id=None):
        # When training in parallel, we need to sepcify an id to allow different thread
        # to write to different log file instead of a single file. Otherwise, it will cause
        # race condition
        print(f"handling {id}")
        if id:
            stdout_log_path = os.path.join(log_folder, f'model.train.stdout.{id}.log')
            stderr_log_path = os.path.join(log_folder, f'model.train.stderr.{id}.log')
        else:
            stdout_log_path = os.path.join(log_folder, 'model.train.stdout.log')
            stderr_log_path = os.path.join(log_folder, 'model.train.stderr.log')
        with Tee(stdout_path=stdout_log_path, stderr_path=stderr_log_path):
            return self._train(
                input_train_data_folder=input_train_data_folder,
                output_model_folder=output_model_folder,
                log_folder=log_folder)
        
    def train_distributed(self, input_train_data_folder, output_model_folder, log_folder):
        stdout_log_path = os.path.join(log_folder, 'model.train.dist.stdout.log')
        stderr_log_path = os.path.join(log_folder, 'model.train.dist.stderr.log')
        with Tee(stdout_path=stdout_log_path, stderr_path=stderr_log_path):
            return self._train_distributed(
                input_train_data_folder=input_train_data_folder,
                output_model_folder=output_model_folder,
                log_folder=log_folder)

    def generate(self, input_train_data_folder, input_model_folder,
                 output_syn_data_folder, log_folder):
        stdout_log_path = os.path.join(log_folder, 'model.generate.stdout.log')
        stderr_log_path = os.path.join(log_folder, 'model.generate.stderr.log')
        with Tee(stdout_path=stdout_log_path, stderr_path=stderr_log_path):
            return self._generate(
                input_train_data_folder=input_train_data_folder,
                input_model_folder=input_model_folder,
                output_syn_data_folder=output_syn_data_folder,
                log_folder=log_folder)
