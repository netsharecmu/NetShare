import inspect
import netshare.ray as ray

from .model_manager import ModelManager


@ray.remote(scheduling_strategy="SPREAD", max_calls=1)
def _train_model(create_new_model, config, input_train_data_folder,
                 output_model_folder, log_folder):
    model = create_new_model(config)
    model.train(
        input_train_data_folder=input_train_data_folder,
        output_model_folder=output_model_folder,
        log_folder=log_folder)
    return True


@ray.remote(scheduling_strategy="SPREAD", max_calls=1)
def _generate_data(create_new_model, config, input_train_data_folder,
                   input_model_folder, output_syn_data_folder, log_folder):
    config["given_data_attribute_flag"] = False
    config["save_without_chunk"] = True
    model = create_new_model(config)
    model.generate(
        input_train_data_folder=input_train_data_folder,
        input_model_folder=input_model_folder,
        output_syn_data_folder=output_syn_data_folder,
        log_folder=log_folder)
    return True


class DGModelManager(ModelManager):

    def _train(self, input_train_data_folder, output_model_folder, log_folder,
               create_new_model, model_config):
        print(f"{self.__class__.__name__}.{inspect.stack()[0][3]}")
        ray.get(_train_model.remote(
            create_new_model=create_new_model,
            config=model_config,
            input_train_data_folder=input_train_data_folder,
            output_model_folder=output_model_folder,
            log_folder=log_folder))
        return True

    def _generate(self, input_train_data_folder, input_model_folder,
                  output_syn_data_folder, log_folder, create_new_model,
                  model_config):
        print(f"{self.__class__.__name__}.{inspect.stack()[0][3]}")
        ray.get(_generate_data.remote(
            create_new_model=create_new_model,
            config=model_config,
            input_train_data_folder=input_train_data_folder,
            input_model_folder=input_model_folder,
            output_syn_data_folder=output_syn_data_folder,
            log_folder=log_folder))
        return True
