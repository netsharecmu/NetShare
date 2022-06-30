# fixed time + interarrival=True
# DP=False, Pretrain=True
# DG-wisc: CAIDA, DC, WISC
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
        "result_root_folder": "../results/results_sigcomm2022_fidelity_plain",
        "ignored_keys_for_folder_name": ["extra_checkpoint_freq", "epoch_checkpoint_freq", "max_flow_len", "num_chunks", "epoch", "self_norm", "num_cores", "sn_mode", "scale", "dataset", "skip_chunk0_train", "pretrain_dir"]
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

        "sleep_time_check_finish": 60,
        "sleep_time_launch_cmd": 5,
        "conda_virtual_env": "NetShare",

    },

    "test_config": [
        {
            "dataset": dict_alias_data["ugr16"],
            "max_flow_len": [dict_alias_maxFlowLen["ugr16"]],
            "num_chunks": [len(dict_alias_data["ugr16"])],
            "iteration": [400000],
            "run": [0],
            "sample_len": [1],
            "extra_checkpoint_freq": [25000],
            "epoch_checkpoint_freq": [1000],

            "pretrain_non_dp": [False],
            "pretrain_non_dp_reduce_time": [None],

            "pretrain_dp": [False],

            # DP
            "dp_noise_multiplier": [None],
            "dp_l2_norm_clip": [None],

            # fine-tuning/pretrain
            # "restore": [False],
            "pretrain_dir": [None],

            "skip_chunk0_train": [False]
        },

        {
            "dataset": dict_alias_data["cidds"],
            "max_flow_len": [dict_alias_maxFlowLen["cidds"]],
            "num_chunks": [len(dict_alias_data["cidds"])],
            "iteration": [400000],
            "run": [0],
            "sample_len": [5, 10, 25, 50],
            "extra_checkpoint_freq": [25000],
            "epoch_checkpoint_freq": [1000],

            "pretrain_non_dp": [False],
            "pretrain_non_dp_reduce_time": [None],

            "pretrain_dp": [False],

            # DP
            "dp_noise_multiplier": [None],
            "dp_l2_norm_clip": [None],

            # fine-tuning/pretrain
            # "restore": [False],
            "pretrain_dir": [None],

            "skip_chunk0_train": [False]

        },

        {
            "dataset": dict_alias_data["ton"],
            "max_flow_len": [dict_alias_maxFlowLen["ton"]],
            "num_chunks": [len(dict_alias_data["ton"])],
            "iteration": [400000],
            "run": [0],
            "sample_len": [5, 10, 25, 50, 100],
            "extra_checkpoint_freq": [25000],
            "epoch_checkpoint_freq": [1000],

            "pretrain_non_dp": [False],
            "pretrain_non_dp_reduce_time": [None],

            "pretrain_dp": [False],

            # DP
            "dp_noise_multiplier": [None],
            "dp_l2_norm_clip": [None],

            # fine-tuning/pretrain
            # "restore": [False],
            "pretrain_dir": [None],

            "skip_chunk0_train": [False]

        },

        {
            "dataset": dict_alias_data["caida"],
            "max_flow_len": [dict_alias_maxFlowLen["caida"]],
            "num_chunks": [len(dict_alias_data["caida"])],
            "iteration": [80000],
            "run": [0],
            "sample_len": [5, 10, 25, 50, 100],
            # "sample_len": [50],
            "extra_checkpoint_freq": [5000],
            "epoch_checkpoint_freq": [1000],

            "pretrain_non_dp": [False],
            "pretrain_non_dp_reduce_time": [None],

            "pretrain_dp": [False],

            # DP
            "dp_noise_multiplier": [None],
            "dp_l2_norm_clip": [None],

            # fine-tuning/pretrain
            # "restore": [False],
            "pretrain_dir": [None],

            # In case chunk0 is trained somewhere else (not with other chunks)
            # NOTE: CHUNK0 must have ckpt(s) in this case
            "skip_chunk0_train": [False]
        },

        {
            "dataset": dict_alias_data["dc"],
            "max_flow_len": [dict_alias_maxFlowLen["dc"]],
            "num_chunks": [len(dict_alias_data["dc"])],
            "iteration": [80000],
            "run": [0],
            "sample_len": [5, 10, 25, 50, 100],
            "extra_checkpoint_freq": [5000],
            "epoch_checkpoint_freq": [1000],

            "pretrain_non_dp": [False],
            "pretrain_non_dp_reduce_time": [None],

            "pretrain_dp": [False],

            # DP
            "dp_noise_multiplier": [None],
            "dp_l2_norm_clip": [None],

            # fine-tuning/pretrain
            # "restore": [False],
            "pretrain_dir": [None],

            "skip_chunk0_train": [False],

        },

        {
            "dataset": dict_alias_data["ca"],
            "max_flow_len": [dict_alias_maxFlowLen["ca"]],
            "num_chunks": [len(dict_alias_data["ca"])],
            "iteration": [160000],
            "run": [0],
            "sample_len": [5, 10, 25, 50, 100],
            "extra_checkpoint_freq": [10000],
            "epoch_checkpoint_freq": [1000],

            "pretrain_non_dp": [False],
            "pretrain_non_dp_reduce_time": [None],

            "pretrain_dp": [False],

            # DP
            "dp_noise_multiplier": [None],
            "dp_l2_norm_clip": [None],

            # fine-tuning/pretrain
            # "restore": [False],
            "pretrain_dir": [None],

            "skip_chunk0_train": [False]

        },
    ]

}