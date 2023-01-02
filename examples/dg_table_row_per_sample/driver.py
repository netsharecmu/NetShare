from netshare import GeneratorV2

if __name__ == "__main__":
    # configuration file
    generator = GeneratorV2(config="config_example_wiki.json")

    # `work_folder` should not exist o/w an overwrite error will be thrown.
    # Please set the `worker_folder` as *absolute path*
    # if you are using Ray with multi-machine setup
    # since Ray has bugs when dealing with relative paths.
    generator.train_and_generate(work_folder="~/results/wiki")
