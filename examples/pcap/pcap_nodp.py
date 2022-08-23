import netshare.ray as ray
from netshare import Generator
from config_io import Config

if __name__ == '__main__':
    ray.config.enabled = True
    ray.init(address="auto")

    generator = Generator(config="config_example_pcap_nodp.json")
    generator.train_and_generate(work_folder='results/test')

    ray.shutdown()
