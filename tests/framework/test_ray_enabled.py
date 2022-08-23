import netshare.ray as ray
from netshare import Generator
# from tests.framework._config import config
from config_io import Config

if __name__ == '__main__':
    ray.config.enabled = True
    ray.init(address="auto")

    config = Config.load_from_file("tests/framework/config_demo.json")

    # config = Config.load_from_file("tests/framework/config_demo.json")
    generator = Generator(config=config)
    # generator.train(work_folder='/nfs/datafuel/results/test')
    generator.generate(work_folder='/nfs/datafuel/results/test')

    ray.shutdown()
