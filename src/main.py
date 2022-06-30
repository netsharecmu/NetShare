import sys, configparser, json, subprocess, time, argparse, datetime
import importlib
import os, re, copy, random, warnings
import tensorflow as tf

from multiprocessing import Process

# https://stackoverflow.com/questions/5228158/cartesian-product-of-a-dictionary-of-lists
import itertools

from gan.util import dict_product, chunks, load_config, load_measurers, configs2configsgroup, get_configid_from_kv, wait_for_chunk0

def kill_all_jobs(measurer_IPs):
    for measurer_ip in measurer_IPs:
        # cmd = "sudo ssh -o StrictHostKeyChecking=no root@{} \"sudo pkill -f generate.py &\""
        cmd = "sudo ssh -o StrictHostKeyChecking=no root@{} \"sudo pkill python3 &\""
        # cmd = "sudo ssh -o StrictHostKeyChecking=no root@{} \"sudo pkill -f 'python3 train.py configs_config4.json' &\""
        # cmd = "sudo ssh -o StrictHostKeyChecking=no root@{} \"sudo pkill -f 'python3 generate' &\""
        cmd = cmd.format(measurer_ip)
        print(cmd)

        subprocess.Popen(cmd, stderr=sys.stdout.fileno(), shell=True)
        time.sleep(1)

    print("Finish killing all measurers' jobs...")


def kill_partial_jobs(config_file, configs, dict_configIdx_measureIP):
    config_idx_to_kill = []
    for config_idx, config in enumerate(configs):
        if ("dc" in config["result_folder"]) and ("epochid-0" not in config["result_folder"]):
            config_idx_to_kill.append(config_idx)

    print("Number of configs to be killed:", len(config_idx_to_kill))
    for config_idx in config_idx_to_kill:
        measurer_idx = dict_configIdx_measureIP[config_idx][0]
        measurer_IP = dict_configIdx_measureIP[config_idx][1]
        print("Config idx: {}, measurer_idx: {}, measurer_IP: {}".format(config_idx, measurer_idx, measurer_IP))

        # ssh_command = "ps -ef | grep 'python3 train.py configs_%s.json %s' | grep -v grep | awk '{print \$2}' | xargs -r sudo kill -9 &"%(config_file, config_idx)
        # cmd = "sudo ssh -o StrictHostKeyChecking=no root@{} \"{}\"".format(measurer_IP, ssh_command)
        # print(cmd)

        cmd = "rm -rf {}".format(configs[config_idx]["result_folder"])
        print(cmd)

        subprocess.Popen(cmd, stderr=sys.stdout.fileno(), shell=True)
        time.sleep(1)
    

# CAUTION: cleanup result folder for each config, used when rerun required
# TO BE REPLACED BY RESTORE MECHANISM
def cleanup_result_folder(configs):
    print("cleaning up result folders...")
    for config in configs:
        result_folder = config["result_folder"]

        # print(result_folder)

        cmd = "rm -rf {}".format(result_folder)
        subprocess.Popen(cmd, stderr=sys.stdout.fileno(), shell=True)
        time.sleep(0.1)


def cleanup_generated_samples_folder(configs):
    print("cleaning up generated samples folders...")
    for config in configs:
        result_folder = config["result_folder"]

        # print(result_folder)

        cmd = "rm -rf {}".format(os.path.join(result_folder, "generated_samples"))
        subprocess.Popen(cmd, stderr=sys.stdout.fileno(), shell=True)
        time.sleep(0.1)


def check_epoch_time(configs, dict_configIdx_measureIP):
    remaining_configs_ids = []
    remaining_time_list = []

    num_finished_configs = 0
    for config_idx, config in enumerate(configs):
        if os.path.exists(os.path.join(config["result_folder"], "time.txt")):
            with open(os.path.join(config["result_folder"], "time.txt"), 'r') as f:
                # print("config_id: {}, measurer_id: {}, measurer_IP: {}".format(config_idx, dict_configIdx_measureIP[config_idx][0], dict_configIdx_measureIP[config_idx][1]))


                lines = f.readlines()

                # print(lines[0].strip())
                # print(lines[-1].strip())

                start_time = datetime.datetime.strptime(lines[0].strip().split("starts: ")[1], '%Y-%m-%d %H:%M:%S.%f')

                if "starts" in lines[-1]:
                    end_time = datetime.datetime.strptime(lines[-1].strip().split("starts: ")[1], '%Y-%m-%d %H:%M:%S.%f')

                elif "ends" in lines[-1]:
                    end_time = datetime.datetime.strptime(lines[-1].strip().split("ends: ")[1], '%Y-%m-%d %H:%M:%S.%f')

                time_elapsed = end_time - start_time

                # print("elapsed time:", time_elapsed)

                # TODO: check epoch == last line epoch?
                last_epoch = int(lines[-1].strip().split()[1]) + 1

                if "ends" in lines[-1] and last_epoch == int(config["epoch"]):
                    num_finished_configs += 1
                    # print()
                else:
                    remaining_configs_ids.append(config_idx)
                    print("config_id: {}, measurer_id: {}, measurer_IP: {}".format(config_idx, dict_configIdx_measureIP[config_idx][0], dict_configIdx_measureIP[config_idx][1]))

                    print(lines[0].strip())
                    print(lines[-1].strip())
                    print("elapsed time:", time_elapsed)

                    est_remaining_time = float(int(config["epoch"]) - last_epoch) / float(last_epoch) * time_elapsed
                    print("estimated remaining time:", est_remaining_time)

                    remaining_time_list.append(est_remaining_time)

                    # print(config)

                    print()

    print("# of total configs:", len(configs))
    print("# of finished configs:", num_finished_configs)
    print("# of running configs:", len(configs) - num_finished_configs)

    remaining_time_list = sorted(remaining_time_list)

    # print("remaining_time_list:", remaining_time_list)

    return remaining_configs_ids

  
def main(args):
    # dynamically import lib
    # https://stackoverflow.com/questions/41678073/import-class-from-module-dynamically
    config = getattr(importlib.import_module("configs."+args.config_file), 'config')

    configs = load_config(config)
    # randomly shuffle the configurations to make sure they are evenly distributed to different measurers
    random.seed(42)
    random.shuffle(configs)
    
    configs, config_group_list = configs2configsgroup(configs, args.generation)
    config_json_file = "configs_{}.json".format(args.config_file)
    fout = open(config_json_file, "w")
    json.dump(configs, fout)
    fout.close()

    for config in configs:
        os.makedirs(config["result_folder"], exist_ok=True)
        os.makedirs(config["eval_root_folder"], exist_ok=True)

    print("# of configs:", len(configs))

    # measurer_IPs = load_measurers(os.path.join("../", "measurer_ini", args.measurer_file))
    measurer_IPs = load_measurers(os.path.join("measurer_ini", args.measurer_file))
    measurer_IPs = measurer_IPs[args.measurer_start_index:]
    print("# of measurers:", len(measurer_IPs))

    configs_idx_split = list(chunks([i for i in range(len(configs))], len(measurer_IPs)))

    dict_configIdx_measureIP = {}

    for measurer_idx, config_idxs in enumerate(configs_idx_split):
        for config_idx in config_idxs:
            dict_configIdx_measureIP[config_idx] = (measurer_idx, measurer_IPs[measurer_idx])

    if args.check_remaining_processes:
        for measurer_ip in measurer_IPs:
            cmd = "sudo ssh -o StrictHostKeyChecking=no root@{} \"ps aux | grep python3 | wc -l \""
            cmd = cmd.format(measurer_ip)
            print(measurer_ip)

            subprocess.Popen(cmd, stderr=sys.stdout.fileno(), shell=True)
            time.sleep(1)

    if args.check_epoch_time:
        remaining_configs_ids = check_epoch_time(configs, dict_configIdx_measureIP)

        # print("remaining configs ids:", remaining_configs_ids)

    if args.kill_all_jobs:
        kill_all_jobs(measurer_IPs)
    
    if args.kill_partial_jobs:
        kill_partial_jobs(args.config_file, configs, dict_configIdx_measureIP)

    if args.cleanup_result_folder:
        cleanup_result_folder(configs)

    if args.cleanup_generated_samples_folder:
        cleanup_generated_samples_folder(configs)


    ################################ Train #####################################
    if args.measurement:
        sub_python_file = "train.py"
        procs = []
        for config_group_id, config_group in enumerate(config_group_list):
            print("Config group {}: DP: {}, pretrain: {}".format(config_group_id, config_group["dp"], config_group["pretrain"]))
            config_ids = config_group["config_ids"]
            if config_group["dp"] == False and config_group["pretrain"] == True:
                chunk0_idx = config_ids[0]
                if configs[chunk0_idx]["skip_chunk0_train"] == True:
                    print("Skipping chunk0 training...")
                else:
                    print("Start launching chunk0 experiments...")
                    # launch chunk0 first
                    config_idx = config_ids[0]
                    measurer_ip = dict_configIdx_measureIP[config_idx][1]
                    log_file = os.path.join(configs[config_idx]["result_folder"], "worker_train.log")

                    cmd = "sudo ssh -o StrictHostKeyChecking=no root@{} \"source ~/anaconda3/etc/profile.d/conda.sh && conda activate {} && cd {} &&  python3 {} {} {} \" > {} 2>&1 &"
                    cmd = cmd.format(measurer_ip, configs[config_idx]["conda_virtual_env"], configs[config_idx]["src_dir"], sub_python_file, config_json_file, config_idx, log_file)
                    print(cmd)

                    subprocess.Popen(cmd, stderr=sys.stdout.fileno(), shell=True)
                    time.sleep(configs[config_idx]["sleep_time_launch_cmd"])
                    print("Finish launching chunk0 experiments ...")
                    print("Start waiting for chunk0 from config_group_id {} experiments finished ...".format(config_group_id))

                proc = Process(
                    target=wait_for_chunk0, 
                    args=(
                        config_group_id,
                        configs,
                        config_ids,
                        dict_configIdx_measureIP,
                        sub_python_file,
                        config_json_file))
                procs.append(proc)
                proc.start()

            else:
                print("Launching all chunks experiments...")
                for config_idx in config_ids:
                    measurer_ip = dict_configIdx_measureIP[config_idx][1]
                    log_file = os.path.join(configs[config_idx]["result_folder"], "worker_train.log")

                    cmd = "sudo ssh -o StrictHostKeyChecking=no root@{} \"source ~/anaconda3/etc/profile.d/conda.sh && conda activate {} && cd {} &&  python3 {} {} {} \" > {} 2>&1 &"
                    cmd = cmd.format(measurer_ip, configs[config_idx]["conda_virtual_env"], configs[config_idx]["src_dir"], sub_python_file, config_json_file, config_idx, log_file)
                    print(cmd)

                    subprocess.Popen(cmd, stderr=sys.stdout.fileno(), shell=True)
                    time.sleep(configs[config_idx]["sleep_time_launch_cmd"])
            
            print()
        
        for proc in procs:
            proc.join()

    ################################## Generation #############################

    elif args.generation:
        time_start_generation = time.time()
        # remove any existing eval_root_folder
        # AND make new empty eval_root_folder
        for config_idx, config in enumerate(configs):
            cmd = "rm -rf {}".format(config["eval_root_folder"])
            print(cmd)
            subprocess.Popen(cmd, stderr=sys.stdout.fileno(), shell=True)
            time.sleep(1)
            os.makedirs(config["eval_root_folder"], exist_ok=True)
        print("Finish cleaning up exisiting eval_root_folder ...\n\n")
        time.sleep(2)

        # generate_attr.py
        print("Start generating attributes ...")
        sub_python_file = "generate_attr.py"
        for config_idx, config in enumerate(configs):
            measurer_ip = dict_configIdx_measureIP[config_idx][1]
            log_file = os.path.join(configs[config_idx]["result_folder"], "worker_generate_attr.log")

            cmd = "sudo ssh -o StrictHostKeyChecking=no root@{} \"source ~/anaconda3/etc/profile.d/conda.sh &&  conda activate {} && cd {} &&  python3 {} {} {} \" > {} 2>&1 &"
            cmd = cmd.format(measurer_ip, configs[config_idx]["conda_virtual_env"], configs[config_idx]["src_dir"], sub_python_file, config_json_file, config_idx, log_file)
            print(cmd)

            subprocess.Popen(cmd, stderr=sys.stdout.fileno(), shell=True)
            time.sleep(configs[config_idx]["sleep_time_launch_cmd"])
        

        # wait for generating attributes to finish
        while True:
            is_generate_attr_finish = True
            num_total_task = 0
            num_remaining_task = 0
            for config_idx, config in enumerate(configs):
                if not os.path.exists(os.path.join(
                    config["eval_root_folder"],
                    "attr_raw",
                    "chunk_id-{}.npz".format(config["chunk_id"])
                )):
                    print("Generating attributes not finished: {}".format(
                        os.path.join(
                            config["eval_root_folder"],
                            "attr_raw",
                            "chunk_id-{}.npz".format(config["chunk_id"])
                        )))

                    is_generate_attr_finish = False
                    num_remaining_task += 1
                
                num_total_task += 1
            
            if is_generate_attr_finish == False:
                print("Generating attributes is not finished! {}/{} tasks remaining...".format(num_remaining_task, num_total_task))
            else:
                print("Generating attributes is finished!")
                break
            time.sleep(configs[config_idx]["sleep_time_check_finish"])
        
        print("Finish generating attributes")

        # merge_attr.py
        # wait for merge_attr to finish
        print("Start merging attributes ...")
        sub_python_file = "merge_attr.py"
        for config_group in config_group_list:
            chunk0_idx = config_group["config_ids"][0]
            eval_root_folder = configs[chunk0_idx]["eval_root_folder"]
            measurer_ip = dict_configIdx_measureIP[chunk0_idx][1]

            log_file = os.path.join(eval_root_folder, "worker_merge_attr.log")
            cmd = "sudo ssh -o StrictHostKeyChecking=no root@{} \"source ~/anaconda3/etc/profile.d/conda.sh &&  conda activate {} && cd {} &&  python3 {} {} {} {} {} \" > {} 2>&1 &"
            # TODO: CHANGE WORD2VEC_SIZE AND PCAP_INTERARRIVAL TO READ FROM CONFIGS
            cmd = cmd.format(measurer_ip, configs[config_idx]["conda_virtual_env"], configs[config_idx]["src_dir"], sub_python_file, os.path.join(eval_root_folder, "attr_raw"), 10, 1, len(config_group["config_ids"]), log_file)
            print(cmd)

            subprocess.Popen(cmd, stderr=sys.stdout.fileno(), shell=True)
            time.sleep(configs[chunk0_idx]["sleep_time_launch_cmd"])
        
        while True:
            is_merge_attr_finish = True
            num_total_task = 0
            num_remaining_task = 0
            for config_idx, config in enumerate(configs):
                if not os.path.exists(os.path.join(
                    config["eval_root_folder"],
                    "attr_clean",
                    "chunk_id-{}.npz".format(config["chunk_id"])
                )):
                    is_merge_attr_finish = False
                    num_remaining_task += 1
                num_total_task += 1
            
            if is_merge_attr_finish == False:
                print("Merging attributes is not finished! {}/{} tasks remaining...".format(num_remaining_task, num_total_task))
            else:
                print("Merging attributes is finished!")
                break
            time.sleep(configs[config_idx]["sleep_time_check_finish"])
        
        # generate features given attribute
        print("Start generating features given attributes ...")
        sub_python_file = "generate_givenattr.py"
        for config_idx, config in enumerate(configs):
            measurer_ip = dict_configIdx_measureIP[config_idx][1]
            log_file = os.path.join(configs[config_idx]["result_folder"], "worker_generate_givenattr.log")

            cmd = "sudo ssh -o StrictHostKeyChecking=no root@{} \"source ~/anaconda3/etc/profile.d/conda.sh &&  conda activate {} && cd {} &&  python3 {} {} {} \" > {} 2>&1 &"
            cmd = cmd.format(measurer_ip, configs[config_idx]["conda_virtual_env"], configs[config_idx]["src_dir"], sub_python_file, config_json_file, config_idx, log_file)
            print(cmd)

            subprocess.Popen(cmd, stderr=sys.stdout.fileno(), shell=True)
            time.sleep(configs[config_idx]["sleep_time_launch_cmd"])

        # wait for generating feature given attr
        while True:
            is_generate_givenattr_finish = True
            num_total_task = 0
            num_remaining_task = 0
            for config_idx, config in enumerate(configs):
                iteration_range = list(range(config["extra_checkpoint_freq"]-1,
                                    config["iteration"],
                                    config["extra_checkpoint_freq"]))
                for iteration_id in iteration_range:
                    if not os.path.exists(os.path.join(
                        config["result_folder"],
                        "checkpoint",
                        "iteration_id-{}".format(iteration_id)
                    )):
                        continue
                    if not os.path.exists(os.path.join(
                        config["eval_root_folder"],
                        "syn_dfs",
                        "chunk_id-{}".format(config["chunk_id"]),
                        "syn_df_iteration_id-{}.csv".format(iteration_id)
                    )):
                        # print("{} not finished!".format(os.path.join(
                        #     config["eval_root_folder"],
                        #     "syn_dfs",
                        #     "chunk_id-{}".format(config["chunk_id"]),
                        #     "syn_df_iteration_id-{}.csv".format(iteration_id)
                        # )))
                        is_generate_givenattr_finish = False
                        num_remaining_task += 1
                    
                    num_total_task += 1

            if is_generate_givenattr_finish == False:
                print("Generating features given attributes is not finished! {}/{} tasks remaining...".format(num_remaining_task, num_total_task))
            else:
                print("Generating features given attributes is finished")
                break
            time.sleep(configs[config_idx]["sleep_time_check_finish"])
        
        time_end_generation = time.time()
        elapsed_time_generation = (time_end_generation - time_start_generation)/60.0
        print("Generation takes {} minutes...".format(elapsed_time_generation))


        ########################################################################
        ########################################################################
        ########################################################################

        # combine synthetic dfs from different chunks
        # (and convert to pcap if applicable)
        # print("Start combining synthetic dfs from different chunks ...")
        # sub_python_file = "merge_syndf.py"
        # for config_group in config_group_list:
        #     chunk0_idx = config_group["config_ids"][0]
        #     eval_root_folder = configs[chunk0_idx]["eval_root_folder"]
        #     measurer_ip = dict_configIdx_measureIP[chunk0_idx][1]
            
        #     log_file = os.path.join(eval_root_folder, "worker_merge_syndf.log")
        #     cmd = "sudo ssh -o StrictHostKeyChecking=no root@{} \"source ~/anaconda3/etc/profile.d/conda.sh &&  conda activate {} && cd {} && python3 {} {} {} \" > {} 2>&1 &"

        #     cmd = cmd.format(measurer_ip, configs[config_idx]["conda_virtual_env"], configs[config_idx]["src_dir"], sub_python_file, os.path.join(eval_root_folder, "syn_dfs"), config["num_chunks"], log_file)
        #     print(cmd)

        #     subprocess.Popen(cmd, stderr=sys.stdout.fileno(), shell=True)
        #     time.sleep(2)

                
                



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # must provide these params
    parser.add_argument('--config_file', type=str)
    parser.add_argument('--measurer_file', type=str)

    parser.add_argument('--measurer_start_index', type=int, default=0)

    '''
    measurement/generation/evaluation: run the three arguments sequentially
    '''

    # measurement: train the model
    parser.add_argument('--measurement', action='store_true', default=False)

    # generation
    parser.add_argument('--generation', action='store_true', default=False)

    # test flags
    # ReLU --> leakyReLU
    parser.add_argument('--leakyrelu', action='store_true', default=False)

    parser.add_argument('--metadataGenONLY', action='store_true', default=False)

    '''
    functional
    '''
    parser.add_argument('--check_remaining_processes', action='store_true', default=False)
    parser.add_argument('--check_epoch_time', action='store_true', default=False)

    # DANGER ZONE
    parser.add_argument('--kill_all_jobs', action='store_true', default=False)
    parser.add_argument('--kill_partial_jobs', action='store_true', default=False)
    parser.add_argument('--cleanup_result_folder', action='store_true', default=False)
    parser.add_argument('--cleanup_generated_samples_folder', action='store_true', default=False)

    args = parser.parse_args()

    main(args)
