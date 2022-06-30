# Created on Dec 13, 2021
# For preprocessing different types of data using 

import pickle, ipaddress, copy, time, more_itertools, os, math, json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

from tqdm import tqdm
from gensim.models import Word2Vec, word2vec
from sklearn import preprocessing

from util import *
from field import *
from output import *
from word2vec_embedding import get_vector
from preprocess_helper import countList2cdf, continuous_list_flag, netflow_flag_str2int

# IP preprocess
def IP_int2str(IP_int):
    return str(ipaddress.ip_address(IP_int))

def IP_str2int(IP_str):
    return int(ipaddress.ip_address(IP_str))

def IPs_int2str(IPs_int):
    return [IP_int2str(i) for i in IPs_int]

def IPs_str2int(IPs_str):
    return [IP_str2int(i) for i in IPs_str]

# Optimistic 
# def flatten_data_feature(data_feature, data_gen_flag):


# type: fixed_size, fixed_time
# n_instances: number of processes you would like to parallel
# time_start/time_end: used for creating non private-data dependent intervals
def df2epochs(args, big_raw_df, file_type="PCAP", split_type="fixed_size", n_instances=10, eps=1e-5):
    if file_type == "PCAP":
        time_col_name = "time"
    elif file_type == "UGR16" or file_type == "CIDDS" or file_type == "TON":
        time_col_name = "ts"

    # sanity sort
    big_raw_df = big_raw_df.sort_values(time_col_name)

    dfs = []
    if split_type == "fixed_size":
        epoch_size = math.ceil(big_raw_df.shape[0] / n_instances)
        for epoch_id in range(n_instances):
            df_epoch = big_raw_df.iloc[epoch_id*epoch_size:((epoch_id+1)*epoch_size)]
            dfs.append(df_epoch)
        return dfs, epoch_size
            
    elif split_type == "fixed_time":
        # time_evenly_spaced = np.linspace(big_raw_df[time_col_name].min(), big_raw_df[time_col_name].max(), num=n_instances+1)
        time_evenly_spaced = np.linspace(args.start_time, args.end_time, num=n_instances+1)
        time_evenly_spaced[-1] *= (1+eps)

        epoch_time = (args.end_time - args.start_time) / n_instances
        
        for epoch_id in range(n_instances):
            df_epoch = big_raw_df[(big_raw_df[time_col_name] >= time_evenly_spaced[epoch_id]) & (big_raw_df[time_col_name] < time_evenly_spaced[epoch_id+1])]
            if len(df_epoch) == 0:
                print("Raw epoch_id: {}, empty df_epoch!".format(epoch_id))
                # continue
            dfs.append(df_epoch)
        return dfs, epoch_time

    else:
        raise ValueError("Unknown split type")
    


# global_max_flow_len: augment/align for fine-tuning
def split_per_epoch(args, fields, df_per_epoch, embed_model, global_max_flow_len, num_epochs, epoch_id, flowkeys_epochidx=None, encode_IP=False):
    # normalize time by per-epoch min/max
    if args.file_type == "PCAP":
        if args.pcap_interarrival == True:
            gk = df_per_epoch.groupby(["srcip", "dstip", "srcport", "dstport", "proto"])
            flow_start_list = []
            interarrival_within_flow_list = []
            for group_name, df_group in gk:
                flow_start_list.append(df_group.iloc[0]["time"])
                interarrival_within_flow_list += ([0.0]+list(np.diff(df_group["time"])))
            # fields["flow_start"].min_x = float(min(flow_start_list))
            # fields["flow_start"].max_x = float(max(flow_start_list))
            fields["flow_start"].min_x = float(df_per_epoch["time"].min())
            fields["flow_start"].max_x = float(df_per_epoch["time"].max())
            fields["interarrival_within_flow"].min_x = args.min_interarrival_within_flow*10**6
            fields["interarrival_within_flow"].max_x = args.max_interarrival_within_flow*10**6
        else:
            fields["time"].min_x = float(df_per_epoch["time"].min())
            fields["time"].max_x = float(df_per_epoch["time"].max())
    elif args.file_type == "UGR16" or args.file_type == "CIDDS" or args.file_type == "TON":
        if args.netflow_interarrival == True:
            gk = df_per_epoch.groupby(["srcip", "dstip", "srcport", "dstport", "proto"])
            flow_start_list = []
            interarrival_within_flow_list = []
            for group_name, df_group in gk:
                flow_start_list.append(df_group.iloc[0]["ts"])
                interarrival_within_flow_list += ([0.0]+list(np.diff(df_group["ts"])))
            # fields["flow_start"].min_x = float(min(flow_start_list))
            # fields["flow_start"].max_x = float(max(flow_start_list))
            fields["flow_start"].min_x = float(df_per_epoch["ts"].min())
            fields["flow_start"].max_x = float(df_per_epoch["ts"].max())
            fields["interarrival_within_flow"].min_x = args.min_interarrival_within_flow*10**6
            fields["interarrival_within_flow"].max_x = args.max_interarrival_within_flow*10**6
        else:
            fields["ts"].min_x = float(df_per_epoch["ts"].min())
            fields["ts"].max_x = float(df_per_epoch["ts"].max())

    split_name = args.split_name
    word2vec_vecSize = args.word2vec_vecSize
    file_type = args.file_type
    
    if "multiepoch_dep" in split_name and flowkeys_epochidx == None:
        raise ValueError("Cross-epoch mechanism enabled, cross-epoch flow stats not provided!")
    
    metadata = ["srcip", "dstip", "srcport", "dstport", "proto"]
    gk = df_per_epoch.groupby(by=metadata)
    
    data_attribute = []
    data_feature = []
    data_gen_flag = []

    num_flows_startFromThisEpoch = 0

    gk_idx = 0
    for group_name, df_group in tqdm(gk):
        # RESET INDEX TO MAKE IT START FROM ZERO
        df_group = df_group.reset_index(drop=True)

        # skip flows whose length > global_max_flow_len
        if len(df_group) > global_max_flow_len:
            continue

        if args.partial_flow == True:
            if (gk_idx+1) > args.partial_flow_until:
                break
            gk_idx += 1

        attr_per_row = []
        feature_per_row = []
        data_gen_flag_per_row = []

        # metadata
        # word2vec
        if encode_IP == True:
            attr_per_row += list(get_vector(embed_model, str(group_name[0]), norm_option=True))
            attr_per_row += list(get_vector(embed_model, str(group_name[1]), norm_option=True))
        # bitwise
        else:
            attr_per_row += fields["srcip"].normalize(group_name[0])
            attr_per_row += fields["dstip"].normalize(group_name[1])

        attr_per_row += list(get_vector(embed_model, str(group_name[2]), norm_option=True))
        attr_per_row += list(get_vector(embed_model, str(group_name[3]), norm_option=True))
        attr_per_row += list(get_vector(embed_model, str(group_name[4]), norm_option=True))

        # pcap: flow start time
        if file_type == "PCAP" and args.pcap_interarrival == True:
            attr_per_row.append(fields["flow_start"].normalize(df_group.iloc[0]["time"]))
        
        if (file_type == "UGR16" or file_type == "CIDDS" or file_type == "TON") and args.netflow_interarrival == True:
            attr_per_row.append(fields["flow_start"].normalize(df_group.iloc[0]["ts"]))

        # cross-epoch generation
        if "multiepoch_dep" in split_name:
            if str(group_name) in flowkeys_epochidx: # sanity check
                # flow starts from this epoch
                if flowkeys_epochidx[str(group_name)][0] == epoch_id:
                    attr_per_row += fields["startFromThisEpoch"].normalize(1.0)
                    num_flows_startFromThisEpoch += 1

                    for i in range(num_epochs):
                        if i in flowkeys_epochidx[str(group_name)]:
                            attr_per_row += fields["epoch_{}".format(i)].normalize(1.0)
                        else:
                            attr_per_row += fields["epoch_{}".format(i)].normalize(0.0)

                # flow does not start from this epoch
                else:
                    attr_per_row += fields["startFromThisEpoch"].normalize(0.0)
                    if split_name == "multiepoch_dep_v1":
                        for i in range(num_epochs):
                            attr_per_row += fields["epoch_{}".format(i)].normalize(0.0)

                    elif split_name == "multiepoch_dep_v2":
                        for i in range(num_epochs):
                            if i in flowkeys_epochidx[str(group_name)]:
                                attr_per_row += fields["epoch_{}".format(i)].normalize(1.0)
                            else:
                                attr_per_row += fields["epoch_{}".format(i)].normalize(0.0)

        data_attribute.append(attr_per_row)

        # measurement
        if file_type == "PCAP" and args.pcap_interarrival == True:
            interarrival_per_flow_list = [0.0] + list(np.diff(df_group["time"]))
        if (file_type == "UGR16" or file_type == "CIDDS" or file_type == "TON") and args.netflow_interarrival == True:
            interarrival_per_flow_list = [0.0] + list(np.diff(df_group["ts"]))

        for row_index, row in df_group.iterrows():
            if file_type == "PCAP":
                if args.pcap_interarrival == True:
                    timeseries_per_step = [fields["interarrival_within_flow"].normalize(interarrival_per_flow_list[row_index]), row["pkt_len"]]
                else:
                    timeseries_per_step = [fields["time"].normalize(row["time"]), row["pkt_len"]]

                if args.full_IP_header == True:
                    for field in ["tos", "id", "flag", "off", "ttl"]:
                        field_normalize = fields[field].normalize(row[field])
                        if isinstance(field_normalize, list):
                            timeseries_per_step += field_normalize
                        else:
                            timeseries_per_step.append(field_normalize)
            
            elif file_type == "UGR16" or file_type == "CIDDS" or file_type == "TON":
                if args.netflow_interarrival == True:
                    timeseries_per_step = [fields["interarrival_within_flow"].normalize(interarrival_per_flow_list[row_index]), row["td"]]
                else:
                    timeseries_per_step = [fields["ts"].normalize(row["ts"]), row["td"]]

                timeseries_per_step.append(row["pkt"])
                timeseries_per_step.append(row["byt"])
                if file_type == "TON":
                    timeseries_per_step += fields["label"].normalize(row["label"])
                timeseries_per_step += fields["type"].normalize(row["type"])

            feature_per_row.append(timeseries_per_step)
            data_gen_flag_per_row.append(1.0)
        
        # Append to global_max_flow_len
        # feature_dim = len(timeseries_per_step)
        # for i in range(global_max_flow_len - len(df_group)):
        #     feature_per_row.append([0.0]*feature_dim)
            # data_gen_flag_per_row.append(0.0)

        data_feature.append(feature_per_row)
        data_gen_flag.append(data_gen_flag_per_row)

    data_attribute = np.asarray(data_attribute)
    data_feature = np.asarray(data_feature)
    data_gen_flag = np.asarray(data_gen_flag)

    print("data_attribute: {}, {}GB in memory".format(np.shape(data_attribute), data_attribute.size*data_attribute.itemsize/1024/1024/1024))
    print("data_feature: {}, {}GB in memory".format(np.shape(data_feature), data_feature.size*data_feature.itemsize/1024/1024/1024))
    print("data_gen_flag: {}, {}GB in memory".format(np.shape(data_gen_flag), data_gen_flag.size*data_gen_flag.itemsize/1024/1024/1024))

    data_attribute_output = []
    data_feature_output = []

    if encode_IP == True:
        for flow_key in ["srcip", "dstip"]:
            for i in range(word2vec_vecSize):
                data_attribute_output.append(fields["{}_{}".format(flow_key, i)].getOutputType())
    else:
        data_attribute_output += fields["srcip"].getOutputType()
        data_attribute_output += fields["dstip"].getOutputType()
    for flow_key in ["srcport", "dstport", "proto"]:
        for i in range(word2vec_vecSize):
            data_attribute_output.append(fields["{}_{}".format(flow_key, i)].getOutputType())
        
    if args.pcap_interarrival == True or args.netflow_interarrival == True:
        data_attribute_output.append(fields["flow_start"].getOutputType())
    
    if "multiepoch_dep" in split_name:
        data_attribute_output.append(fields["startFromThisEpoch"].getOutputType())
        for i in range(num_epochs):
            data_attribute_output.append(fields["epoch_{}".format(i)].getOutputType())

    if file_type == "PCAP":
        if args.pcap_interarrival == True:
            field_list = ["interarrival_within_flow"]
        else:
            field_list = ["time"]

        # for field in ["time", "pkt_len", "tos", "id", "flag", "off", "ttl"]:
        if args.full_IP_header == True:
            field_list += ["pkt_len", "tos", "id", "flag", "off", "ttl"]
        else:
            field_list += ["pkt_len"]

        for field in field_list:
            field_output = fields[field].getOutputType()
            if isinstance(field_output, list):
                data_feature_output += field_output
            else:
                data_feature_output.append(field_output)
    
    if file_type == "UGR16" or file_type == "CIDDS" or file_type == "TON":
        if args.netflow_interarrival == True:
            field_list = ["interarrival_within_flow"]
        else:
            field_list = ["ts"]

        if file_type == "UGR16" or file_type == "CIDDS":
            field_list += ["td", "pkt", "byt", "type"]
        else:
            field_list += ["td", "pkt", "byt", "label", "type"]

        for field in field_list:
            field_output = fields[field].getOutputType()
            if isinstance(field_output, list):
                data_feature_output += field_output
            else:
                data_feature_output.append(field_output)

    print("data_attribute_output:", len(data_attribute_output))
    print("data_feature_output:", len(data_feature_output))

    return data_attribute, data_feature, data_gen_flag, data_attribute_output, data_feature_output, fields


def main(args):
    '''load data and embed model'''
    df = pd.read_csv(os.path.join(args.src_dir, args.src_csv))

    # log-transform raw data
    if args.file_type == "UGR16" or args.file_type == "CIDDS" or args.file_type == "TON":
        df["td"] = np.log(1+df["td"])
        df["pkt"] = np.log(1+df["pkt"])
        df["byt"] = np.log(1+df["byt"])
    
    ## TODO: TEMP FOR CIDDS; CHANGE LATER
    if args.file_type == "CIDDS":
        df.drop('type', axis=1, inplace=True)
        df = df.replace({"label":{"attacker": "blacklist", "victim": "blacklist", "normal": "background"}})
        df = df.rename(columns={"label":"type"})


    embed_model = Word2Vec.load(os.path.join(args.src_dir, "word2vec_vecSize_{}.model".format(args.word2vec_vecSize)))
    print("Processing dataset {} ...".format(args.src_dir))
    print(df.shape)


    if args.norm_option == 0:
        NORM_OPTION = Normalization.ZERO_ONE
        norm_opt_str = "ZERO_ONE"
    elif args.norm_option == 1:
        NORM_OPTION = Normalization.MINUSONE_ONE
        norm_opt_str = "MINUSONE_ONE"
    else:
        raise ValueError("Invalid normalization option!")


    '''create and normalize fields'''
    fields = {}

    if args.encode_IP == False:
        fields["srcip"] = BitField(
            name="srcip", 
            num_bits=32
        )

        fields["dstip"] = BitField(
            name="dstip", 
            num_bits=32
        )

    for i in range(args.word2vec_vecSize):
        if args.encode_IP == True:
            fields["srcip_{}".format(i)] = ContinuousField(
                name="srcip_{}".format(i),
                norm_option=Normalization.MINUSONE_ONE,
                dim_x=1
            )
            fields["dstip_{}".format(i)] = ContinuousField(
                name="dstip_{}".format(i),
                norm_option=Normalization.MINUSONE_ONE,
                dim_x=1
            )
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
    
    if "multiepoch_dep" in args.split_name:
        fields["startFromThisEpoch"] = DiscreteField(
            name="startFromThisEpoch", 
            choices=[0.0, 1.0]
        )

        for epoch_id in range(args.n_instances):
            fields["epoch_{}".format(epoch_id)] = DiscreteField(
            name="epoch_{}".format(epoch_id), 
            choices=[0.0, 1.0]
            )
    
    if args.file_type == "PCAP" or args.file_type == "FBFLOW":
        if args.pcap_interarrival == True:
            fields["flow_start"] = ContinuousField(
                name="flow_start",
                norm_option=NORM_OPTION
            )

            fields["interarrival_within_flow"] = ContinuousField(
                name="interarrival_within_flow",
                norm_option=NORM_OPTION
            )


        else:
            fields["time"] = ContinuousField(
                name="time",
                # min_x=float(df["time"].min()),
                # max_x=float(df["time"].max()),
                norm_option=NORM_OPTION
            )
            # df["time"] = fields["time"].normalize(df["time"])

        fields["pkt_len"] = ContinuousField(
            name="pkt_len",
            # min_x=float(df["pkt_len"].min()),
            # max_x=float(df["pkt_len"].max()),
            min_x=20.0,
            max_x=1500.0,
            norm_option=NORM_OPTION
        )
        df["pkt_len"] = fields["pkt_len"].normalize(df["pkt_len"])

        if args.file_type == "PCAP" and args.full_IP_header == True:
            # fields["tos"] = BitField(
            #     name="tos", 
            #     num_bits=8
            # )

            fields["tos"] = ContinuousField(
                name="tos",
                # min_x=float(df["tos"].min()),
                # max_x=float(df["tos"].max()),
                min_x=0.0,
                max_x=255.0,
                norm_option=NORM_OPTION
            )

            fields["id"] = ContinuousField(
                name="id",
                # min_x=float(df["id"].min()),
                # max_x=float(df["id"].max()),
                min_x=0.0,
                max_x=65535.0,
                norm_option=NORM_OPTION
            )

            fields["flag"] = DiscreteField(
                name="flag", 
                choices=[0, 1, 2]
            )

            fields["off"] = ContinuousField(
                name="off",
                # min_x=float(df["off"].min()),
                # max_x=float(df["off"].max()),
                min_x=0.0,
                max_x=8191.0, # 13 bits
                norm_option=NORM_OPTION
            )

            fields["ttl"] = ContinuousField(
                name="ttl",
                # min_x=float(df["ttl"].min()),
                # max_x=float(df["ttl"].max()),
                min_x=1.0, # TTL less than 1 will be rounded up to 1; TTL=0 will be dropped.
                max_x=255.0,
                norm_option=NORM_OPTION
            )
    
    if args.file_type == "UGR16" or args.file_type == "CIDDS" or args.file_type == "TON":
        if args.netflow_interarrival == True:
            fields["flow_start"] = ContinuousField(
                name="flow_start",
                norm_option=NORM_OPTION
            )

            fields["interarrival_within_flow"] = ContinuousField(
                name="interarrival_within_flow",
                norm_option=NORM_OPTION
            )

        else:
            fields["ts"] = ContinuousField(
                name="ts",
                # min_x=float(df["te"].min()),
                # max_x=float(df["te"].max()),
                norm_option=NORM_OPTION
            )
            # df["te"] = fields["te"].normalize(df["te"])

        fields["td"] = ContinuousField(
            name="td",
            # min_x=float(df["td"].min()),
            # max_x=float(df["td"].max()),
            min_x=math.log(1+0.0),
            max_x=math.log(1+600.0), # 15 minutues for timeout settings
            norm_option=NORM_OPTION
        )
        df["td"] = fields["td"].normalize(df["td"])

        fields["pkt"] = ContinuousField(
            name="pkt",
            # min_x=float(df["pkt"].min()),
            # max_x=float(df["pkt"].max()),
            min_x=math.log(1+1),
            max_x=math.log(1+10**6), # 1M packets per flow
            norm_option=NORM_OPTION
        )
        df["pkt"] = fields["pkt"].normalize(df["pkt"])
        
        fields["byt"] = ContinuousField(
            name="byt",
            # min_x=float(df["byt"].min()),
            # max_x=float(df["byt"].max()),
            min_x=math.log(1+20), # 20 bytes * 1 packet
            max_x=math.log(1+1500*10**6), # 1500 bytes * 1M packets
            norm_option=NORM_OPTION
        )
        df["byt"] = fields["byt"].normalize(df["byt"])

        if args.file_type == "TON":
            fields["label"] = DiscreteField(
                name="label", 
                choices=list(set(df["label"]))
            )

        fields["type"] = DiscreteField(
            name="type", 
            choices=list(set(df["type"]))
        )
    
    print(df.head())
    print("Number of fields:", len(fields))


    '''define metadata and measurement'''
    # metadata = ["srcip", "dstip", "srcport", "dstport", "proto", "srchostprefix", "dsthostprefix", "srcrack", "dstrack", "srcpod", "dstpod", "intercluster", "interdatacenter"]
    metadata = ["srcip", "dstip", "srcport", "dstport", "proto"]
    measurement = list(set(df.columns) - set(metadata))
    print("metadata:", metadata)
    print("measurement:", measurement)
    gk = df.groupby(by=metadata)
    flow_len_list = sorted(gk.size().values, reverse=True)
    print(flow_len_list[:10])
    x, cdf = countList2cdf(flow_len_list)
    plt.plot(x, cdf, linewidth=2)
    plt.xlabel("# of packets per flow")
    plt.ylabel("cdf")
    # plt.xscale('log')
    plt.savefig(os.path.join(args.src_dir, "raw_pkts_per_flow.png"), bbox_inches="tight", dpi=300)
    


    '''generating cross-epoch flow stats'''
    # split big df to epochs
    print("Using {}".format(args.df2epochs))

    if args.df2epochs == "fixed_time":
        if (args.start_time is None) or (args.end_time is None):
            raise ValueError("Privacy mode: start/end time must be specified!")
        
        if (args.global_max_flow_len is None):
            raise ValueError("Privacy mode: global_max_flow_len must be specified!")

        if (args.min_interarrival_within_flow is None) or (args.max_interarrival_within_flow is None):
            raise ValueError("Privacy mode: interarrival within flow must be provided!")


    df_epochs, _tmp_sizeortime = df2epochs(args, df, file_type=args.file_type,split_type=args.df2epochs, n_instances=args.n_instances)

    df_epoch_cnt_validation = 0
    for epoch_id, df_epoch in enumerate(df_epochs):
        print("Epoch_id: {}, # of pkts/records: {}".format(epoch_id, len(df_epoch)))
        df_epoch_cnt_validation += len(df_epoch)
    print("df_epoch_cnt_validation:", df_epoch_cnt_validation)


    if args.df2epochs == "fixed_size":
        epoch_size = _tmp_sizeortime
        print("Epoch size: {}".format(epoch_size))
    elif args.df2epochs == "fixed_time":
        epoch_time = _tmp_sizeortime
        print("Epoch time: {} seconds".format(epoch_time / (10**6)))
    else:
        raise ValueError("Unknown df2epochs type!")




    flowkeys_epochidx_file = os.path.join(args.src_dir, "flowkeys_idxlist.json")
    # if os.path.exists(flowkeys_epochidx_file):
    #     print("load flowkey-epoch list...")
    #     with open(flowkeys_epochidx_file, 'r') as f:
    #         flowkeys_epochidx = json.load(f)
    # else:
    print("compute flowkey-epoch list from scratch...")
    flow_epochid_keys = {}

    for epoch_id, df_epoch in enumerate(df_epochs):
        if len(df_epoch) == 0:
            print("Epoch_id {} empty! Skipping ...".format(epoch_id))
            flow_epochid_keys[epoch_id] = []
        else:
            gk = df_epoch.groupby(metadata)
            flow_keys = list(gk.groups.keys())
            flow_keys = list(map(str, flow_keys))
            flow_epochid_keys[epoch_id] = flow_keys
    
    # key: flow key
    # value: [a list of appeared epoch idx]
    flowkeys_epochidx = {}

    for epoch_id, flowkeys in flow_epochid_keys.items():
        print("processing epoch {}/{}, # of flows: {}".format(epoch_id+1, len(df_epochs), len(flowkeys)))
        for k in flowkeys:
            if k not in flowkeys_epochidx:
                flowkeys_epochidx[k] = []
            flowkeys_epochidx[k].append(epoch_id)
    
    with open(flowkeys_epochidx_file, 'w') as f:
        json.dump(flowkeys_epochidx, f)




    num_non_continuous_flows = 0
    num_flows_cross_epoch  = 0
    flowkeys_epochlen_list = []
    for flowkey, epochidx in flowkeys_epochidx.items():
        flowkeys_epochlen_list.append(len(epochidx))
        if not continuous_list_flag(epochidx):
            num_non_continuous_flows += 1
        if len(epochidx) > 1:
            num_flows_cross_epoch += 1
    
    print("# of total flows:", len(flowkeys_epochlen_list))
    print("# of total flows (sanity check):", len(df.groupby(metadata)))
    print("# of flows cross epoch (of total flows): {} ({}%)".format(num_flows_cross_epoch, float(num_flows_cross_epoch)/len(flowkeys_epochlen_list)*100))
    print("# of non-continuous flows:", num_non_continuous_flows)
    x, cdf = countList2cdf(flowkeys_epochlen_list)
    plt.clf()
    plt.plot(x, cdf, linewidth=2)
    plt.xlabel("# of epochs per flow")
    plt.ylabel("cdf")
    # plt.xscale('log')
    plt.savefig(os.path.join(args.src_dir, "raw_flow_numepochs.png"), bbox_inches="tight", dpi=300)

    # global_max_flow_len for consistency between epochs
    per_chunk_flow_len_agg = []
    max_flow_lens = []
    for epoch_id, df_epoch in enumerate(df_epochs):
        # corner case: skip for empty df_epoch
        if len(df_epoch) == 0:
            continue
        gk_epoch = df_epoch.groupby(by=metadata)
        max_flow_lens.append(max(gk_epoch.size().values))
        per_chunk_flow_len_agg += list(gk_epoch.size().values)
        print("epoch_id: {}, max_flow_len: {}".format(epoch_id, max(gk_epoch.size().values)))
    # global_max_flow_len = max(max_flow_lens)
    print("global max flow len:", args.global_max_flow_len)
    print("Top 10 per-chunk flow length:", sorted(per_chunk_flow_len_agg)[-10:])



    '''prepare DG training data for each epoch'''
    for epoch_id, df_epoch in tqdm(enumerate(df_epochs)):
        # skip empty df_epoch: corner case
        if len(df_epoch) == 0:
            print("Epoch_id {} empty! Skipping ...".format(epoch_id))
            continue

        # INTERNAL TEST
        if args.partial_epoch == True:
            if (epoch_id+1) > args.partial_epoch_until:
                break
        
        print("\nEpoch_id:", epoch_id)
        data_attribute, data_feature, data_gen_flag, data_attribute_output, data_feature_output, fields_per_epoch = split_per_epoch(
            args=args,
            fields=copy.deepcopy(fields),
            df_per_epoch=df_epoch.copy(),
            embed_model=embed_model,
            global_max_flow_len=args.global_max_flow_len,
            num_epochs=args.n_instances,
            epoch_id=epoch_id,
            flowkeys_epochidx=flowkeys_epochidx,
            encode_IP=args.encode_IP
        )

        data_out_dir = os.path.join(args.src_dir, "split-{},epochid-{},maxFlowLen-{},Norm-{},vecSize-{},df2epochs-{},interarrival-{},fullIPHdr-{},encodeIP-{}".format(args.split_name, epoch_id, args.global_max_flow_len, norm_opt_str, args.word2vec_vecSize, args.df2epochs, args.pcap_interarrival | args.netflow_interarrival, args.full_IP_header, args.encode_IP))
        os.makedirs(data_out_dir, exist_ok=True)

        df_epoch.to_csv(os.path.join(data_out_dir, "raw.csv"), index=False)
        # np.savez(os.path.join(data_out_dir, "data_train.npz"),       
        #     data_feature=data_feature, 
        #     data_attribute=data_attribute, 
        #     data_gen_flag=data_gen_flag)

        ################## WRITE DATA IN A ROW-BASED WAY #####################
        num_rows = data_attribute.shape[0]
        os.makedirs(os.path.join(data_out_dir, "data_train_npz"), exist_ok=True)
        gt_lengths = []
        for row_id in range(num_rows):
            gt_lengths.append(sum(data_gen_flag[row_id]))
            np.savez(os.path.join(data_out_dir, "data_train_npz", "data_train_{}.npz".format(row_id)),       
                data_feature=data_feature[row_id], 
                data_attribute=data_attribute[row_id], 
                data_gen_flag=data_gen_flag[row_id],
                global_max_flow_len=[args.global_max_flow_len],
                # row_id=[row_id]
                )
        np.save(os.path.join(data_out_dir, "gt_lengths"), gt_lengths)

        ######################################################################
    
        with open(os.path.join(data_out_dir, "data_feature_output.pkl"), 'wb') as fout:
            pickle.dump(data_feature_output, fout)
        with open(os.path.join(data_out_dir, "data_attribute_output.pkl"), 'wb') as fout:
            pickle.dump(data_attribute_output, fout)
        with open(os.path.join(data_out_dir, "fields.pkl"), 'wb') as fout:
            pickle.dump(fields_per_epoch, fout)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', type=str, default="../data/fbflow/Cluster-A-00001")
    parser.add_argument('--src_csv', type=str, default="raw.csv")
    parser.add_argument('--word2vec_vecSize', type=int, default=32) 
    parser.add_argument('--file_type', type=str, default="PCAP")
    parser.add_argument('--norm_option', type=int, default=0)

    # multiepoch_ind: multiple epochs, no addi metadata
    # multiepoch_dep_v1: cross-epoch, <five tuples, 1, 1, 0, 0, ... 1>
    #                                 <five tuples, 0, 0, 0, 0, ... 0>
    # multiepoch_dep_v2: cross-epoch, <five tuples, 1, 1, 0, 0, ... 1>
    #                                 <five tuples, 0, 1, 0, 0, ... 1>
    parser.add_argument('--split_name', type=str, default="multiepoch_dep_v1")

    # how to split chunks: fixed_time (for privacy), fixed_size
    parser.add_argument('--df2epochs', type=str, default="fixed_time")
    parser.add_argument('--start_time', type=float, default=None)
    parser.add_argument('--end_time', type=float, default=None)
    parser.add_argument('--global_max_flow_len', type=int, default=None)
    parser.add_argument('--min_interarrival_within_flow', type=float, default=None)
    parser.add_argument('--max_interarrival_within_flow', type=float, default=None)

    parser.add_argument('--n_instances', type=int, default=10, help="number of instances/chunks for running in parallel")

    # use interarrival instead of absolute packet arrival time
    parser.add_argument('--pcap_interarrival', action='store_true', default=False)

    # use interarrival for flow start time
    parser.add_argument('--netflow_interarrival', action='store_true', default=False)

    # pcap_full_header=False: <five tuples, timestamp, pkt_len>
    # pcap_full_header=True: full IP header + port number from tcp/udp header
    parser.add_argument('--full_IP_header', action='store_true', default=False)

    # False (default): IPs with bitwise rep
    # True: encode IPs with word2vec
    parser.add_argument('--encode_IP', action='store_true', default=False)

    # INTERNAL TEST
    parser.add_argument('--partial_epoch', action='store_true', default=False)
    parser.add_argument('--partial_epoch_until', type=int, default=1)
    parser.add_argument('--partial_flow', action='store_true', default=False)
    parser.add_argument('--partial_flow_until', type=int, default=1000)

    args = parser.parse_args()
    main(args) 