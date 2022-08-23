import netshare.ray as ray
from netshare import Generator
# from tests.framework._config import config
from config_io import Config

if __name__ == '__main__':
    ray.config.enabled = False
    ray.init(address="auto")

    config = Config.load_from_file("tests/framework/config.json")
    generator = Generator(config=config)
    generator.train_and_generate(work_folder='results/test')
