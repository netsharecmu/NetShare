import netshare.ray as ray
import os
import torch
from concurrent.futures import ThreadPoolExecutor

@ray.remote(scheduling_strategy="SPREAD", max_calls=1)
def _launch_one_chunk_training(
        create_new_model, configs, config_idx, input_train_data_folder,
        output_model_folder, log_folder, assigned_device="cuda" if torch.cuda.is_available() else "cpu"):
    configs[config_idx].device = assigned_device
    model = create_new_model(configs[config_idx])
    if assigned_device != "cuda" or  assigned_device != "cpu":
        obj = model.train(input_train_data_folder, output_model_folder, log_folder, assigned_device)
    else:
        obj = model.train(input_train_data_folder, output_model_folder, log_folder)
    return obj

@ray.remote(scheduling_strategy="SPREAD", max_calls=1)
def _launch_one_chunk_training_with_ddp(
        create_new_model, configs, config_idx, input_train_data_folder,
        output_model_folder, log_folder):
    model = create_new_model(configs[config_idx])
    obj = model.train_distributed(input_train_data_folder, output_model_folder, log_folder)
    return obj

def _launch_other_chunks_training(
        create_new_model, configs, config_ids, input_train_data_folder,
        output_model_folder, log_folder):
    chunk0_idx = config_ids[0]
    if configs[chunk0_idx]["skip_chunk0_train"] and configs[chunk0_idx][
            "pretrain_dir"] is None:
        raise ValueError(
            "Skipping chunk0 training but chunk0 has no available ckpt!"
            "Please move ckpts into the corresponding folder.")
    # if torch.cuda.is_available:
    # else:
    num_gpus = torch.cuda.device_count()
    if  num_gpus >= 1:
        num_total_jobs = len(config_ids) - 1
        job_assigner = [[] for _ in range(num_gpus)]
        # construct a job queue for each gpu and assign the config id into that
        for i in range(1, len(config_ids)):
            job_assigner[(i-1)%num_gpus].append(i)

        def train_by_cuda_id(gpu_id):
            cuda_id = f"cuda:{gpu_id}"
            objs = []
            if len(job_assigner[gpu_id]) == 0:
                return objs
            
            for config_idx in job_assigner[gpu_id]:
                objs.append(
                    _launch_one_chunk_training.remote(
                        create_new_model,
                        configs,
                        config_idx,
                        input_train_data_folder,
                        output_model_folder,
                        log_folder,
                        cuda_id)
                )
            return objs

        futures = []
        with ThreadPoolExecutor(max_workers=64) as executor:
            for i in range(num_gpus):
                futures.append(executor.submit(
                    train_by_cuda_id, i))
        results = [fut.result() for fut in futures]
        print("------")
    else:
        objs = []
        for config_idx in config_ids[1:]:
            # sanity check
            if not os.path.exists(configs[config_idx]["pretrain_dir"]):
                raise ValueError(
                    f"Pretrain_dir {configs[config_idx]['pretrain_dir']} does not exist!")
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
        create_new_model, configs, config_ids, input_train_data_folder,
        output_model_folder, log_folder):
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


@ray.remote(scheduling_strategy="SPREAD", runtime_env={"env_vars": {"some": "key"}})
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
            if torch.cuda.is_available() and configs[config_idx]["ddp_first_chunk"]:
                result = ray.get(
                    _launch_one_chunk_training_with_ddp.remote(
                        create_new_model,
                        configs,
                        config_idx,
                        input_train_data_folder,
                        output_model_folder,
                        log_folder))
            else:
                result = ray.get(
                    _launch_one_chunk_training.remote(
                        create_new_model,
                        configs,
                        config_idx,
                        input_train_data_folder,
                        output_model_folder,
                        log_folder))

            print("Finish launching chunk0 experiments ...")

        if len(configs) > 1:
            print(
                f"Start waiting for other chunks from config_group_id {config_group_id} experiments finished ...")
            results = _launch_other_chunks_training(
                create_new_model,
                configs,
                config_ids,
                input_train_data_folder,
                output_model_folder,
                log_folder)
            print(f"Other chunks from config_group_id {config_group_id} training finished")

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
