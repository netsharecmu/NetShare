import random
import time

from netshare import Generator

random.seed(time.time())

if __name__ == "__main__":
    generator = Generator(
        config="config_example_wiki.json",
        # `work_folder` should not exist o/w an overwrite error will be thrown.
        work_folder="../../results/test_wiki-" + str(random.randint(0, 1000000)),
    )
    generator.train_and_generate()
