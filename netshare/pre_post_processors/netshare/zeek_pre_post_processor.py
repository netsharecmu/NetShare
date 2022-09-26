import pickle
import ipaddress
import copy
import time
import more_itertools
import os
import math
import json
import sys
from attr import attr

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

from tqdm import tqdm
from gensim.models import Word2Vec, word2vec
from sklearn import preprocessing

from ..pre_post_processor import PrePostProcessor
from netshare.utils import Tee, Output, output
from netshare.utils import Normalization
from netshare.utils import ContinuousField, DiscreteField, BitField
from netshare.utils import exec_cmd

# IP preprocess


def IP_int2str(IP_int):
    return str(ipaddress.ip_address(IP_int))


def IP_str2int(IP_str):
    return int(ipaddress.ip_address(IP_str))


def IPs_int2str(IPs_int):
    return [IP_int2str(i) for i in IPs_int]


def IPs_str2int(IPs_str):
    return [IP_str2int(i) for i in IPs_str]


def main(args):
    '''load data and embed model'''
    df = pd.read_csv(os.path.join(args.src_dir, args.src_csv))

    # conver IP from string to integer
    df["srcip"] = IPs_str2int(df["srcip"])
    df["dstip"] = IPs_str2int(df["dstip"])

    # log transform for fields with large range
    for field in ["duration", "orig_bytes", "resp_bytes", "missed_bytes", "orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes"]:
        df[field] = np.log(1+df[field])

    # load word2vec model
    embed_model = Word2Vec.load(os.path.join(
        args.src_dir, "word2vec_vecSize_{}.model".format(args.word2vec_vecSize)))
    print("Processing dataset {} ...".format(args.src_dir))
    print(df.shape)
    print(df.head())

    if args.norm_option == 0:
        NORM_OPTION = Normalization.ZERO_ONE
        norm_opt_str = "ZERO_ONE"
    elif args.norm_option == 1:
        NORM_OPTION = Normalization.MINUSONE_ONE
        norm_opt_str = "MINUSONE_ONE"
    else:
        raise ValueError("Invalid normalization option!")

    # fields rep
    fields = {}

    # bit encodings for IP
    fields["srcip"] = BitField(
        name="srcip",
        num_bits=32
    )
    fields["dstip"] = BitField(
        name="dstip",
        num_bits=32
    )

    # ip2vec for port/proto
    for i in range(args.word2vec_vecSize):
        fields["srcport_{}".format(i)] = ContinuousField(
            name="srcport_{}".format(i),
            norm_option=Normalization.MINUSONE_ONE,
            dim_x=1
        )
        fields["dstport_{}".format(i)] = ContinuousField(
            name="dstport_{}".format(i),
            norm_option=Normalization.MINUSONE_ONE,
            dim_x=1
        )
        fields["proto_{}".format(i)] = ContinuousField(
            name="proto_{}".format(i),
            norm_option=Normalization.MINUSONE_ONE,
            dim_x=1
        )

    # continuous fields
    for field in ["ts", "duration", "orig_bytes", "resp_bytes", "missed_bytes", "orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes"]:
        fields[field] = ContinuousField(
            name=field,
            min_x=float(df[field].min()),
            max_x=float(df[field].max()),
            norm_option=NORM_OPTION
        )
        df[field] = fields[field].normalize(df[field])

    # categorical/discrete fields
    fields["service"] = DiscreteField(
        name="service",
        choices=list(set(df["service"]))
    )
    fields["conn_state"] = DiscreteField(
        name="conn_state",
        choices=["S0", "S1", "SF", "REJ", "S2", "S3", "RSTO",
                 "RSTR", "RSTOS0", "RSTRH", "SH", "SHR", "OTH"]
    )

    # group by five tuples
    metadata = ["srcip", "dstip", "srcport", "dstport", "proto"]
    gk = df.groupby(by=metadata)

    max_flow_len = max(gk.size())
    print("max_flow_len:", max_flow_len)

    data_attribute = []
    data_feature = []
    data_gen_flag = []

    # generate DG-compatible input
    for group_name, df_group in tqdm(gk):
        df_group = df_group.reset_index(drop=True)

        attr_per_row = []
        feature_per_row = []
        data_gen_flag_per_row = []

        # metadata
        attr_per_row += fields["srcip"].normalize(group_name[0])
        attr_per_row += fields["dstip"].normalize(group_name[1])
        attr_per_row += list(get_vector(embed_model,
                             str(group_name[2]), norm_option=True))
        attr_per_row += list(get_vector(embed_model,
                             str(group_name[3]), norm_option=True))
        attr_per_row += list(get_vector(embed_model,
                             str(group_name[4]), norm_option=True))

        data_attribute.append(attr_per_row)

        # measurement
        for row_index, row in df_group.iterrows():
            feature_per_step = []

            # continuous fields
            for field in ["ts", "duration", "orig_bytes", "resp_bytes", "missed_bytes", "orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes"]:
                feature_per_step.append(row[field])

            # discrete fields
            for field in ["service", "conn_state"]:
                feature_per_step += fields[field].normalize(row[field])

            feature_per_row.append(feature_per_step)
            data_gen_flag_per_row.append(1.0)

        # append 0s for alignment
        if len(df_group) < max_flow_len:
            for i in range(max_flow_len - len(df_group)):
                feature_per_row.append([0.0]*len(feature_per_step))
                data_gen_flag_per_row.append(0.0)

        data_feature.append(feature_per_row)
        data_gen_flag.append(data_gen_flag_per_row)

    data_attribute = np.asarray(data_attribute)
    data_feature = np.asarray(data_feature)
    data_gen_flag = np.asarray(data_gen_flag)

    print("data_attribute: {}, {}GB in memory".format(np.shape(
        data_attribute), data_attribute.size*data_attribute.itemsize/1024/1024/1024))
    print("data_feature: {}, {}GB in memory".format(np.shape(data_feature),
          data_feature.size*data_feature.itemsize/1024/1024/1024))
    print("data_gen_flag: {}, {}GB in memory".format(np.shape(
        data_gen_flag), data_gen_flag.size*data_gen_flag.itemsize/1024/1024/1024))

    # attribute/feature output
    data_attribute_output = []
    data_feature_output = []

    data_attribute_output += fields["srcip"].getOutputType()
    data_attribute_output += fields["dstip"].getOutputType()
    for flow_key in ["srcport", "dstport", "proto"]:
        for i in range(args.word2vec_vecSize):
            data_attribute_output.append(
                fields["{}_{}".format(flow_key, i)].getOutputType())

    field_list = ["ts", "duration", "orig_bytes", "resp_bytes", "missed_bytes",
                  "orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes", "service", "conn_state"]
    for field in field_list:
        field_output = fields[field].getOutputType()
        if isinstance(field_output, list):
            data_feature_output += field_output
        else:
            data_feature_output.append(field_output)

    print("data_attribute_output:", len(data_attribute_output))
    print("data_feature_output:", len(data_feature_output))

    # DG-compatible input
    # 1. data_train.npz
    # 2. data_attribute_output
    # 3. data_feature_output
    np.savez(os.path.join(args.src_dir, "data_train.npz"),
             data_feature=data_feature,
             data_attribute=data_attribute,
             data_gen_flag=data_gen_flag)
    with open(os.path.join(args.src_dir, "data_feature_output.pkl"), 'wb') as fout:
        pickle.dump(data_feature_output, fout)
    with open(os.path.join(args.src_dir, "data_attribute_output.pkl"), 'wb') as fout:
        pickle.dump(data_attribute_output, fout)

    # save fields, load at post-process time
    with open(os.path.join(args.src_dir, "fields.pkl"), 'wb') as fout:
        pickle.dump(fields, fout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', type=str, default="../data/conn_log")
    parser.add_argument('--src_csv', type=str, default="raw.csv")
    parser.add_argument('--word2vec_vecSize', type=int, default=10)
    parser.add_argument('--norm_option', type=int, default=0)

    args = parser.parse_args()
    main(args)
