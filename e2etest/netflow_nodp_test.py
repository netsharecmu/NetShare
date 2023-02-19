import pytest
import random
import time
import os.path

from netshare import Generator

random.seed(time.time())    

def test_driver_netflow_nodp():
    randValue = str(random.randint(0, 1000000))
    generator = Generator(
        # configuration file
        config="../examples/netflow/config_example_netflow_nodp.json",
        # `work_folder` should not exist o/w an overwrite error will be thrown.
        work_folder="../results/test_ugr16-" + randValue,
    )
    generator.train_and_generate()
    
    assert os.path.isdir("../results/test_ugr16-" + randValue) == True
