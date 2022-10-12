from netshare import Generator

if __name__ == '__main__':
    # configuration file
    generator = Generator(
        config="zeeklog/config_example_zeeklog.json")

    # `work_folder` should not exist o/w an overwrite error will be thrown.
    # Please set the `worker_folder` as *absolute path*
    # if you are using Ray with multi-machine setup
    # since Ray has bugs when dealing with relative paths.
    generator.visualize(work_folder='../results/vis_test_zeek')
