# fixed time + interarrival=True 
# DG-emulab : ugr16, caida, dc
import os

dict_alias_data = {}
dict_alias_maxFlowLen = {}

for alias in ["ugr16", "cidds", "ton", "caida", "dc", "ca"]:
    dirs = [x[0] for x in os.walk("../data/1M/{}".format(alias))]
    data = []
    for dir in dirs:
        if len(dir.split('/')) == 5 and \
            "df2epochs-fixed_time" in dir and \
            "interarrival-True" in dir:
            dir = dir.replace("../data/", "")
            data.append(dir)
            if alias not in dict_alias_maxFlowLen:
                kvs = dir.split("/")[-1].split(",")
                for item in kvs:
                    k = item.split("-")[0]
                    v = item.split("-")[1]
                    if k == "maxFlowLen":
                        dict_alias_maxFlowLen[alias] = int(v)

    dict_alias_data[alias] = data
    if alias in dict_alias_data:
        print(alias, len(dict_alias_data[alias]))
    if alias in dict_alias_maxFlowLen:
        print(alias, dict_alias_maxFlowLen[alias])

config = {
	"scheduler_config": {
        "result_root_folder": "../results/results_test1_pcap_no_dp",
        "ignored_keys_for_folder_name": ["extra_checkpoint_freq", "epoch_checkpoint_freq", "max_flow_len", "num_chunks", "epoch", "self_norm", "num_cores", "sn_mode", "scale", "dataset"]
    },
	
	"global_config": {
        "batch_size": 100,
        "vis_freq": 100000,
        "vis_num_sample": 5,
        "d_rounds": 5,
        "g_rounds": 1,
        "num_packing": 1,
        "noise": True,
        "attr_noise_type": "normal",
        "feature_noise_type": "normal",
        "rnn_mlp_num_layers": 0,
        "feed_back": False,
        "g_lr": 0.0001,
        "d_lr": 0.0001,
        "d_gp_coe": 10.0,
        "gen_feature_num_layers": 1,
        "gen_feature_num_units": 100,
        "gen_attribute_num_layers": 5,
        "gen_attribute_num_units": 512,
        "disc_num_layers": 5,
        "disc_num_units": 512,
        "initial_state": "random",

        "leaky_relu": False,

        "attr_d_lr": 0.0001,
        "attr_d_gp_coe": 10.0,
        "g_attr_d_coe": 1.0,
        "attr_disc_num_layers": 5,
        "attr_disc_num_units": 512,

        # dir containg the src code
        "src_dir": "/nfs/NetShare/src",
        # "pretrain_dir": None,

        "aux_disc": True,
        "self_norm": False,
        "fix_feature_network": False,
        "debug": False,
        "combined_disc": True,

        "use_gt_lengths": False,
        "use_uniform_lengths": False,

        # profiling
        "num_cores": None,

        # SN
        "sn_mode": None,
        "scale": 1.0,

        # cmd time
        "sleep_time_check_finish": 60,
        "sleep_time_launch_cmd": 5,
        "conda_virtual_env": "NetShare",

    },

    "test_config": [
        {
            "dataset": dict_alias_data["caida"],
            "max_flow_len": [dict_alias_maxFlowLen["caida"]],
            "num_chunks": [len(dict_alias_data["caida"])],
            "iteration": [40],
            "run": [0],
            "sample_len": [10],
            "extra_checkpoint_freq": [10],
            "epoch_checkpoint_freq": [5],

            # pretrain_non_DP: only use for non-DP version
            #   True: the first chunk will be trained first and every following chunk will be trained on this chunk (fewer total CPU hours)
            #   False: every chunk will be trained simultaneously 
            "pretrain_non_dp": [True],
            
            # pretrain_non_dp_reduce_time: only use for non-DP version
            # how much less of time you would like to train for starting the second chunk?
            "pretrain_non_dp_reduce_time": [4.0],

            # pretrain_DP: only use for DP version
            #   True: every chunk will be trained on public data
            #   False: naive DP-SGD, no public data involved
            "pretrain_dp": [False],

            # DP
            "dp_noise_multiplier": [None],
            "dp_l2_norm_clip": [None],

            # fine-tuning/pretrain
            # "restore": [False],
            "pretrain_dir": [None],
        }

    ]

}