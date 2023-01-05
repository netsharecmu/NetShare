import random
import time

from netshare import Generator

random.seed(time.time())

if __name__ == "__main__":
    generator = Generator(
        # configuration file
        config="../examples/netflow/config_example_netflow_nodp.json",
        # `work_folder` should not exist o/w an overwrite error will be thrown.
        work_folder="../results/test_ugr16-" + str(random.randint(0, 1000000)),
    )
    generator.train_and_generate()
