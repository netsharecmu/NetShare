from netshare import GeneratorV2

if __name__ == "__main__":
    generator = GeneratorV2(config="config_example_sensors.json")
    generator.train_and_generate()
