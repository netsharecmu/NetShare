from netshare import Generator

if __name__ == "__main__":
    generator = Generator(config="config_example_sensors.json")
    generator.train_and_generate()
