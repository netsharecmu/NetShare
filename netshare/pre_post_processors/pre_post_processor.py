from abc import ABC, abstractmethod
import os

from netshare.utils import Tee


class PrePostProcessor(ABC):
    def __init__(self, config):
        self._config = config

    @abstractmethod
    def _pre_process(self, input_folder, output_folder, log_folder):
        ...

    @abstractmethod
    def _post_process(self, input_folder, output_folder,
                      pre_processed_data_folder, log_folder):
        ...

    def pre_process(self, input_folder, output_folder, log_folder):
        stdout_log_path = os.path.join(log_folder, 'pre_process.stdout.log')
        stderr_log_path = os.path.join(log_folder, 'pre_process.stderr.log')
        with Tee(stdout_path=stdout_log_path, stderr_path=stderr_log_path):
            return self._pre_process(
                input_folder=input_folder,
                output_folder=output_folder,
                log_folder=log_folder)

    def post_process(self, input_folder, output_folder,
                     pre_processed_data_folder, log_folder):
        stdout_log_path = os.path.join(log_folder, 'post_process.stdout.log')
        stderr_log_path = os.path.join(log_folder, 'post_process.stderr.log')
        with Tee(stdout_path=stdout_log_path, stderr_path=stderr_log_path):
            return self._post_process(
                input_folder=input_folder,
                output_folder=output_folder,
                pre_processed_data_folder=pre_processed_data_folder,
                log_folder=log_folder)
