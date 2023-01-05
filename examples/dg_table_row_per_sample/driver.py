import random

from netshare import Generator

if __name__ == "__main__":
    generator = Generator(
        config="config_example_wiki.json",
        # `work_folder` should not exist o/w an overwrite error will be thrown.
        work_folder="../../results/wiki-" + str(random.randint(0, 1000000)),
    )
    generator.train_and_generate()
