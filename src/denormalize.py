import sys, configparser, json, random, copy, math
import os, pickle
from pathlib import Path
import tensorflow as tf
import numpy as np

from gan import output
sys.modules["output"] = output

from gan import field
sys.modules["field"] = field

from tqdm import tqdm

import matplotlib.pyplot as plt
import pandas as pd
from gensim.models import Word2Vec
from collections import Counter, OrderedDict
import statsmodels.api as sm

sys.path.append("../preprocess")
from embedding_helper import build_annoy_dictionary_word2vec, get_original_obj

random.seed(42)

# WORD2VEC_SIZE=32
ENCODE_IP=False

def last_lvl_folder(folder):
    return str(Path(folder).parents[0])

# ../results/results_sigcomm2022/1M/ugr16/split-multiepoch_dep_v1,epochid-0,maxFlowLen-33,Norm-ZERO_ONE,vecSize-32,df2epochs-fixed_time,interarrival-False,fullIPHdr-False,encodeIP-False/epoch-400,run-0,sample_len-1,self_norm-False,num_cores-None,dp_noise_multiplier-None,dp_l2_norm_clip-None,sn_mode-None,scale-1.0,restore-False,pretrain_dir-None

def denormalize(data_attribute, data_feature, data_gen_flag, config):
    df_list = []

    if "interarrival-False" in config["dataset"]:
        interarrival_flag = False
    elif "interarrival-True" in config["dataset"]:
        interarrival_flag = True

    if "/caida/" in config["dataset"] or "/dc/" in config["dataset"] or "/ca/" in config["dataset"]:
        file_type = "pcap"
    elif "ugr16" in config["dataset"]:
        file_type = "ugr16"
    elif "cidds" in config["dataset"]:
        file_type = "cidds"
    elif "ton" in config["dataset"]:
        file_type = "ton"
    else:
        raise ValueError("Unknown data type! Must be netflow or pcap...")

    split_list = config["result_folder"].split("/")[-2].split(",")
    split_dict = {}
    for kvpair in split_list:
        print(kvpair)
        split_dict[kvpair.split("-")[0]] = kvpair.split("-")[1]
    WORD2VEC_SIZE = int(split_dict["vecSize"])
    
    raw_data_folder = os.path.join("../data", config["dataset"])

    fin = open(os.path.join(raw_data_folder, "fields.pkl"), "rb")
    fields = pickle.load(fin)
    fin.close()

    word2vec_model_path = os.path.join(last_lvl_folder(raw_data_folder), "word2vec_vecSize_{}.model".format(WORD2VEC_SIZE))

    # use big raw df or small df?
    # big_raw_df = pd.read_csv(os.path.join(last_lvl_folder(raw_data_folder), "raw.csv"))
    # ann_port, port_dic, ann_proto, proto_dic = build_annoy_dictionary_word2vec(big_raw_df, word2vec_model_path, WORD2VEC_SIZE, file_type="PCAP", n_trees=100, encode_IP=ENCODE_IP)
    
    '''
    # non-DP case: use per-chunk raw df
    if config["dp_noise_multiplier"] is None:
        print("Non DP case: use per-chunk raw df for word2vec NN search...")
        per_chunk_raw_df = pd.read_csv(os.path.join(raw_data_folder, "raw.csv"))
        ann_port, port_dic, ann_proto, proto_dic = build_annoy_dictionary_word2vec(per_chunk_raw_df, word2vec_model_path, WORD2VEC_SIZE, file_type="PCAP", n_trees=1000, encode_IP=ENCODE_IP)
    # DP case: use df that trains the word2vec model
    else:
        print("DP case: use big public df (for word2vec encoding) for word2vec NN search...")
        if "/caida/" in config["dataset"]:
            word2vec_public_df = pd.read_csv(os.path.join("../data", "word2vec", "caida", "raw.csv"))
        elif "ugr16" in config["dataset"]:
            word2vec_public_df = pd.read_csv(os.path.join("../data", "word2vec", "ugr16", "raw.csv"))
        # TODO: SUPPORT OTHER PRIVATE DATASETS
        else:
            raise ValueError("Only CAIDA and UGR16 are currently supported under DP cases!")
        ann_port, port_dic, ann_proto, proto_dic = build_annoy_dictionary_word2vec(word2vec_public_df, word2vec_model_path, WORD2VEC_SIZE, file_type="PCAP", n_trees=1000, encode_IP=ENCODE_IP)
    '''

    per_chunk_raw_df = pd.read_csv(os.path.join(raw_data_folder, "raw.csv"))
    ann_port, port_dic, ann_proto, proto_dic = build_annoy_dictionary_word2vec(per_chunk_raw_df, word2vec_model_path, WORD2VEC_SIZE, file_type="PCAP", n_trees=1000, encode_IP=ENCODE_IP)
    
    print("Finish building annoy dictionary")

    attr_set = set()

    for i in tqdm(range(np.shape(data_attribute)[0])):
        attr_per_row = data_attribute[i]

        srcip = list(attr_per_row[0:64])
        dstip = list(attr_per_row[64:128])
        # srcport = list(attr_per_row[128:160])
        # dstport = list(attr_per_row[160:192])
        # proto = list(attr_per_row[192:224])
        srcport = list(attr_per_row[128:128+WORD2VEC_SIZE])
        dstport = list(attr_per_row[128+WORD2VEC_SIZE:128+WORD2VEC_SIZE*2])
        proto = list(attr_per_row[128+WORD2VEC_SIZE*2:128+WORD2VEC_SIZE*3])

        srcip = fields["srcip"].denormalize(srcip)
        dstip = fields["dstip"].denormalize(dstip)
        srcport = get_original_obj(ann_port, srcport, port_dic)
        dstport = get_original_obj(ann_port, dstport, port_dic)
        proto = get_original_obj(ann_proto, proto, proto_dic)

        # use flow start + interarrival
        if interarrival_flag == True:
            flow_start_time = fields["flow_start"].denormalize(attr_per_row[128+WORD2VEC_SIZE*3])
            cur_pkt_time = flow_start_time

        # remove duplicated attributes (five tuples)
        if (srcip, dstip, srcport, dstport, proto) in attr_set:
            continue
        attr_set.add((srcip, dstip, srcport, dstport, proto))

        for j in range(np.shape(data_feature)[1]):
            if data_gen_flag[i][j] == 1.0:
                df_per_row = {}

                df_per_row["srcip"] = srcip
                df_per_row["dstip"] = dstip
                df_per_row["srcport"] = srcport
                df_per_row["dstport"] = dstport
                df_per_row["proto"] = proto

                if file_type == "ugr16" or file_type == "cidds" or file_type == "ton":
                    if interarrival_flag == False:
                        df_per_row["ts"] = fields["ts"].denormalize(data_feature[i][j][0])
                    else:
                        if j == 0:
                            df_per_row["ts"] = flow_start_time
                        else:
                            df_per_row["ts"] = cur_pkt_time + fields["interarrival_within_flow"].denormalize(data_feature[i][j][0])
                            cur_pkt_time = df_per_row["ts"]

                    df_per_row["td"] = math.exp(fields["td"].denormalize(data_feature[i][j][1]))-1
                    df_per_row["pkt"] = math.exp(fields["pkt"].denormalize(data_feature[i][j][2]))-1
                    df_per_row["byt"] = math.exp(fields["byt"].denormalize(data_feature[i][j][3]))-1

                    if file_type == "ugr16":
                        df_per_row["type"] = fields["type"].denormalize(data_feature[i][j][4:])
                    elif file_type == "cidds":
                        df_per_row["label"] = fields["label"].denormalize(data_feature[i][j][4:7])
                        df_per_row["type"] = fields["type"].denormalize(data_feature[i][j][7:])
                    elif file_type == "ton":
                        df_per_row["label"] = fields["label"].denormalize(data_feature[i][j][4:6])
                        df_per_row["type"] = fields["type"].denormalize(data_feature[i][j][6:])
                    else:
                        raise ValueError("Unknown netflow type!")
                
                if file_type == "pcap":
                    if interarrival_flag == False:
                        df_per_row["time"] = fields["time"].denormalize(data_feature[i][j][0])
                    else:
                        if j == 0:
                            df_per_row["time"] = flow_start_time
                        else:
                            df_per_row["time"] = cur_pkt_time + fields["interarrival_within_flow"].denormalize(data_feature[i][j][0])
                            cur_pkt_time = df_per_row["time"]
                    df_per_row["pkt_len"] = fields["pkt_len"].denormalize(data_feature[i][j][1])
                    df_per_row["tos"] = fields["tos"].denormalize(data_feature[i][j][2])
                    df_per_row["id"] = fields["id"].denormalize(data_feature[i][j][3])
                    df_per_row["flag"] = fields["flag"].denormalize(list(data_feature[i][j][4:7]))
                    df_per_row["off"] = fields["off"].denormalize(data_feature[i][j][7])
                    df_per_row["ttl"] = fields["ttl"].denormalize(data_feature[i][j][8])


                df_list.append(df_per_row)


    df = pd.DataFrame(df_list)
    print(df.head())

    return df