import netshare.ray as ray
import os


@ray.remote(scheduling_strategy="SPREAD", max_calls=1)
def _launch_one_chunk_training(
        create_new_model, configs, config_idx, input_train_data_folder, output_model_folder, log_folder):
    model = create_new_model(configs[config_idx])
    obj = model.train(input_train_data_folder, output_model_folder, log_folder)
    return obj


def _launch_other_chunks_training(
        create_new_model, configs, config_ids, input_train_data_folder, output_model_folder, log_folder):
    chunk0_idx = config_ids[0]
    if configs[chunk0_idx]["skip_chunk0_train"] and configs[chunk0_idx]["pretrain_dir"] is None:
        raise ValueError(
            "Skipping chunk0 training but chunk0 has no available ckpt!"
            "Please move ckpts into the corresponding folder.")
    objs = []
    for config_idx in config_ids[1:]:
        # sanity check
        if not os.path.exists(configs[config_idx]["pretrain_dir"]):
            raise ValueError("Pretrain_dir {} does not exist!")
        objs.append(
            _launch_one_chunk_training.remote(
                create_new_model,
                configs,
                config_idx,
                input_train_data_folder,
                output_model_folder,
                log_folder))

    results = ray.get(objs)
    return results


def _launch_all_chunks_training(
        create_new_model, configs, config_ids, input_train_data_folder, output_model_folder, log_folder):
    objs = []
    for config_idx in config_ids:
        # sanity check
        if not os.path.exists(configs[config_idx]["pretrain_dir"]):
            raise ValueError("Pretrain_dir {} does not exist!")
        objs.append(
            _launch_one_chunk_training.remote(
                create_new_model,
                configs,
                config_idx,
                input_train_data_folder,
                output_model_folder,
                log_folder))

    results = ray.get(objs)
    return results


@ray.remote(scheduling_strategy="SPREAD")
def _train_specific_config_group(
        create_new_model,
        config_group_id,
        config_group,
        configs,
        input_train_data_folder,
        output_model_folder,
        log_folder):
    print(
        "Config group {}: DP: {}, pretrain: {}".format(
            config_group_id, config_group["dp"], config_group["pretrain"]
        )
    )
    config_ids = config_group["config_ids"]
    if config_group["dp"] == False and config_group["pretrain"] == True:
        chunk0_idx = config_ids[0]
        if configs[chunk0_idx]["skip_chunk0_train"] == True:
            print("Skipping chunk0 training...")
        else:
            print("Start launching chunk0 experiments...")
            # launch first chunk
            config_idx = config_ids[0]
            result = ray.get(
                _launch_one_chunk_training.remote(
                    create_new_model,
                    configs,
                    config_idx,
                    input_train_data_folder,
                    output_model_folder,
                    log_folder))

            print("Finish launching chunk0 experiments ...")
            print(
                "Start waiting for chunk0 from config_group_id {}"
                "experiments finished ...".format(
                    config_group_id
                )
            )
        if len(configs) > 1:
            results = _launch_other_chunks_training(
                create_new_model,
                configs,
                config_ids,
                input_train_data_folder,
                output_model_folder,
                log_folder)
            print("Other chunks training finished")

    else:
        print("Launching all chunks experiments...")
        # Haven't been tested
        results = _launch_all_chunks_training(
            create_new_model,
            configs,
            config_ids,
            input_train_data_folder,
            output_model_folder,
            log_folder)

    return True
