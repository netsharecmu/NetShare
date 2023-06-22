import random
import netshare.ray as ray
from netshare import Generator
import torch

if __name__ == '__main__':
    # Change to False if you would not like to use Ray
    ray.config.enabled = False
    if ray.config.enabled:
        raise Exception("Ray cannot be launched when running on a single node with multiple gpus")
    ray.init(address="auto")

    # configuration file
    generator = Generator(config="config_example_pcap_nodp.json")

    # `work_folder` should not exist o/w an overwrite error will be thrown.
    # Please set the `worker_folder` as *absolute path*
    # if you are using Ray with multi-machine setup
    # since Ray has bugs when dealing with relative paths.
    generator.train(work_folder=f'/lustre/minhao/results/m57-500k')
    generator.generate(work_folder=f'/lustre/minhao/results/m57-500k')
    generator.visualize(work_folder=f'/lustre/minhao/results/m57-500k')

    ray.shutdown()
