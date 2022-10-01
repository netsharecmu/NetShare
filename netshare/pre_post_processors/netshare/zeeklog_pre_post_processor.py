import inspect
import pickle
import ipaddress
import copy
import time
import more_itertools
import os
import math
import json
import sys
import re
from attr import attr

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import netshare.ray as ray

from tqdm import tqdm
from gensim.models import Word2Vec, word2vec
from sklearn import preprocessing
from .util import denormalize, _recalulate_config_ids_in_each_config_group, _merge_syn_df
from .word2vec_embedding import word2vec_train
from .preprocess_helper import df2chunks, continuous_list_flag, split_per_chunk
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


class ZeeklogPrePostProcessor(PrePostProcessor):
    def _pre_process(self, input_folder, output_folder, log_folder):
        print(f"{self.__class__.__name__}.{inspect.stack()[0][3]}")

        # Check if the encode_IP is "bit"
        if self._config["encode_IP"] != "bit":
            raise ValueError(
                "Zeeklog encode_IP hasn't support other types except bit!")

        df = pd.read_csv(input_folder)
        df.to_csv(os.path.join(output_folder, "raw.csv"), index=False)

        # convert IP from string to integer
        df["srcip"] = IPs_str2int(df["srcip"])
        df["dstip"] = IPs_str2int(df["dstip"])

        # log transform for fields with large range
        for field in ["duration", "orig_bytes", "resp_bytes", "missed_bytes",
                      "orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes"]:
            df[field] = np.log(1 + df[field])

        # train/load word2vec model
        # Using DP: pretrained word2vec model exists
        if os.path.exists(os.path.join(
            os.path.dirname(input_folder),
            "word2vec_vecSize_{}.model".format(
                self._config["word2vec_vecSize"])
        )):
            print("Loading pretrained `big` word2vec model...")
            embed_model_name = os.path.join(
                os.path.dirname(input_folder),
                "word2vec_vecSize_{}.model".format(
                    self._config["word2vec_vecSize"])
            )
        else:
            print("Training word2vec model from scratch...")
            embed_model_name = word2vec_train(
                df=df,
                out_dir=output_folder,
                file_type=self._config["dataset_type"],
                encode_IP='bit'
            )
        embed_model = Word2Vec.load(embed_model_name)

        # fields
        if self._config["norm_option"] == 0:
            NORM_OPTION = Normalization.ZERO_ONE
            norm_opt_str = "ZERO_ONE"
        elif self._config["norm_option"] == 1:
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
        for i in range(self._config["word2vec_vecSize"]):
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

        if "multichunk_dep" in self._config["split_name"]:
            fields["startFromThisChunk"] = DiscreteField(
                name="startFromThisChunk",
                choices=[0.0, 1.0]
            )

            for chunk_id in range(self._config["n_chunks"]):
                fields["chunk_{}".format(chunk_id)] = DiscreteField(
                    name="chunk_{}".format(chunk_id),
                    choices=[0.0, 1.0]
                )

        # continuous fields
        for field in ["ts", "duration", "orig_bytes", "resp_bytes", "missed_bytes",
                      "orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes"]:
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

        '''generating cross-chunk flow stats'''
        # split big df to chunks
        print("Using {}".format(self._config["df2chunks"]))
        df_chunks, _tmp_sizeortime = df2chunks(
            big_raw_df=df,
            file_type=self._config["dataset_type"],
            split_type=self._config["df2chunks"],
            n_chunks=self._config["n_chunks"])

        df_chunk_cnt_validation = 0
        for chunk_id, df_chunk in enumerate(df_chunks):
            print("Chunk_id: {}, # of pkts/records: {}".format(
                chunk_id, len(df_chunk)))
            df_chunk_cnt_validation += len(df_chunk)
        print("df_chunk_cnt_validation:", df_chunk_cnt_validation)

        if self._config["df2chunks"] == "fixed_size":
            chunk_size = _tmp_sizeortime
            print("Chunk size: {} records \
                (pkts for PCAP, # of records for NetFlow)".format(
                chunk_size))
        elif self._config["df2chunks"] == "fixed_time":
            chunk_time = _tmp_sizeortime
            print("Chunk time: {} seconds".format(chunk_time / (10**6)))
        else:
            raise ValueError("Unknown df2chunks type!")

        flowkeys_chunkidx_file = os.path.join(
            output_folder, "flowkeys_idxlist.json")
        print("compute flowkey-chunk list from scratch...")
        flow_chunkid_keys = {}
        for chunk_id, df_chunk in enumerate(df_chunks):
            gk = df_chunk.groupby(metadata)
            flow_keys = list(gk.groups.keys())
            flow_keys = list(map(str, flow_keys))
            flow_chunkid_keys[chunk_id] = flow_keys

        # key: flow key
        # value: [a list of appeared chunk idx]
        flowkeys_chunkidx = {}
        for chunk_id, flowkeys in flow_chunkid_keys.items():
            print("processing chunk {}/{}, # of flows: {}".format(
                chunk_id + 1, len(df_chunks), len(flowkeys)))
            for k in flowkeys:
                if k not in flowkeys_chunkidx:
                    flowkeys_chunkidx[k] = []
                flowkeys_chunkidx[k].append(chunk_id)

        with open(flowkeys_chunkidx_file, 'w') as f:
            json.dump(flowkeys_chunkidx, f)

        num_non_continuous_flows = 0
        num_flows_cross_chunk = 0
        flowkeys_chunklen_list = []
        for flowkey, chunkidx in flowkeys_chunkidx.items():
            flowkeys_chunklen_list.append(len(chunkidx))
            if not continuous_list_flag(chunkidx):
                num_non_continuous_flows += 1
            if len(chunkidx) > 1:
                num_flows_cross_chunk += 1

        print("# of total flows:", len(flowkeys_chunklen_list))
        print("# of total flows (sanity check):", len(df.groupby(metadata)))
        print("# of flows cross chunk (of total flows): {} ({}%)".format(
            num_flows_cross_chunk,
            float(num_flows_cross_chunk) / len(flowkeys_chunklen_list) * 100))
        print("# of non-continuous flows:", num_non_continuous_flows)

        # global_max_flow_len for consistency between chunks
        per_chunk_flow_len_agg = []
        max_flow_lens = []
        for chunk_id, df_chunk in enumerate(df_chunks):
            # corner case: skip for empty df_chunk
            if len(df_chunk) == 0:
                continue
            gk_chunk = df_chunk.groupby(by=metadata)
            max_flow_lens.append(max(gk_chunk.size().values))
            per_chunk_flow_len_agg += list(gk_chunk.size().values)
            print("chunk_id: {}, max_flow_len: {}".format(
                chunk_id, max(gk_chunk.size().values)))
        # TODO ???????
        # if not self._config["max_flow_len"]:
        #     global_max_flow_len = max(max_flow_lens)
        if "max_flow_len" not in self._config:
            global_max_flow_len = max(max_flow_lens)
        else:
            global_max_flow_len = self._config["max_flow_len"]
        print("global max flow len:", global_max_flow_len)
        print("Top 10 per-chunk flow length:",
              sorted(per_chunk_flow_len_agg)[-10:])

        '''prepare NetShare training data for each chunk'''
        objs = []
        for chunk_id, df_chunk in tqdm(enumerate(df_chunks)):
            # skip empty df_chunk: corner case
            if len(df_chunk) == 0:
                print("Chunk_id {} empty! Skipping ...".format(chunk_id))
                continue

            print("\nChunk_id:", chunk_id)
            objs.append(split_per_chunk.remote(
                config={**copy.deepcopy(self._config),
                        **copy.deepcopy(self._config)},
                fields=copy.deepcopy(fields),
                df_per_chunk=df_chunk.copy(),
                embed_model=embed_model,
                global_max_flow_len=global_max_flow_len,
                chunk_id=chunk_id,
                flowkeys_chunkidx=flowkeys_chunkidx,
            ))

        objs_output = ray.get(objs)

        # TODO: distribute writing of numpy to the files
        for chunk_id, df_chunk in tqdm(enumerate(df_chunks)):
            data_attribute, data_feature, data_gen_flag, \
                data_attribute_output, data_feature_output, \
                fields_per_epoch = objs_output[chunk_id]
            # TODO: add pre_process multiple configurations
            data_out_dir = os.path.join(
                output_folder,
                f"chunkid-{chunk_id}")
            os.makedirs(data_out_dir, exist_ok=True)

            df_chunk.to_csv(os.path.join(
                data_out_dir, "raw.csv"), index=False)

            num_rows = data_attribute.shape[0]
            os.makedirs(os.path.join(
                data_out_dir, "data_train_npz"), exist_ok=True)
            gt_lengths = []
            for row_id in range(num_rows):
                gt_lengths.append(sum(data_gen_flag[row_id]))
                np.savez(os.path.join(
                    data_out_dir,
                    "data_train_npz", f"data_train_{row_id}.npz"),
                    data_feature=data_feature[row_id],
                    data_attribute=data_attribute[row_id],
                    data_gen_flag=data_gen_flag[row_id],
                    global_max_flow_len=[global_max_flow_len],
                )
            np.save(os.path.join(data_out_dir, "gt_lengths"), gt_lengths)

            with open(os.path.join(
                    data_out_dir,
                    "data_feature_output.pkl"), 'wb') as fout:
                pickle.dump(data_feature_output, fout)
            with open(os.path.join(
                    data_out_dir, "data_attribute_output.pkl"), 'wb') as fout:
                pickle.dump(data_attribute_output, fout)
            with open(os.path.join(
                    data_out_dir, "fields.pkl"), 'wb') as fout:
                pickle.dump(fields_per_epoch, fout)

        return True

    def _post_process(self, input_folder, output_folder, log_folder):
        print(f"{self.__class__.__name__}.{inspect.stack()[0][3]}")
        configs = []

        # Step 1: denormalize to csv

        print("POSTPROCESSOR......")
        print(input_folder)
        paths = [os.path.join(input_folder, p)
                 for p in os.listdir(input_folder)]

        for path in paths:
            feat_raw_path = os.path.join(path, "feat_raw")
            syn_path = os.path.join(path, "syn_dfs")
            os.makedirs(syn_path, exist_ok=True)

            data_files = os.listdir(feat_raw_path)

            for d in data_files:
                data_path = os.path.join(feat_raw_path, d)
                data = np.load(data_path, allow_pickle=True)
                attributes = data["attributes"]
                features = data["features"]
                gen_flags = data["gen_flags"]
                # recover dict from 0-d numpy array
                # https://stackoverflow.com/questions/22661764/storing-a-dict-with-np-savez-gives-unexpected-result
                config = data["config"][()]
                configs.append(config)

                syn_df = denormalize(
                    attributes, features, gen_flags, config)
                chunk_id, iteration_id = re.search(
                    r"chunk_id-(\d+)_iteration_id-(\d+).npz", d).groups()
                print(syn_df.shape)

                save_path = os.path.join(
                    syn_path,
                    "chunk_id-{}".format(chunk_id))
                os.makedirs(save_path, exist_ok=True)
                syn_df.to_csv(
                    os.path.join(
                        save_path,
                        "syn_df_iteration_id-{}.csv".format(iteration_id)),
                    index=False)

            # Step 2: pick the best among hyperparameters/tranining snapshots
            config_group_list = _recalulate_config_ids_in_each_config_group(
                configs)
            work_folder = os.path.dirname(input_folder)
            _merge_syn_df(configs=configs,
                          config_group_list=config_group_list,
                          big_raw_df=pd.read_csv(os.path.join(
                              work_folder, 'pre_processed_data', "raw.csv")),
                          output_syn_data_folder=output_folder
                          )
        return True


def main(args):
    '''load data and embed model'''
    df = pd.read_csv(os.path.join(args.src_dir, args.src_csv))

    # conver IP from string to integer
    df["srcip"] = IPs_str2int(df["srcip"])
    df["dstip"] = IPs_str2int(df["dstip"])

    # log transform for fields with large range
    for field in ["duration", "orig_bytes", "resp_bytes", "missed_bytes",
                  "orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes"]:
        df[field] = np.log(1 + df[field])

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
    for field in ["ts", "duration", "orig_bytes", "resp_bytes", "missed_bytes",
                  "orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes"]:
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
            for field in ["ts", "duration", "orig_bytes", "resp_bytes", "missed_bytes",
                          "orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes"]:
                feature_per_step.append(row[field])

            # discrete fields
            for field in ["service", "conn_state"]:
                feature_per_step += fields[field].normalize(row[field])

            feature_per_row.append(feature_per_step)
            data_gen_flag_per_row.append(1.0)

        # append 0s for alignment
        if len(df_group) < max_flow_len:
            for i in range(max_flow_len - len(df_group)):
                feature_per_row.append([0.0] * len(feature_per_step))
                data_gen_flag_per_row.append(0.0)

        data_feature.append(feature_per_row)
        data_gen_flag.append(data_gen_flag_per_row)

    data_attribute = np.asarray(data_attribute)
    data_feature = np.asarray(data_feature)
    data_gen_flag = np.asarray(data_gen_flag)

    print("data_attribute: {}, {}GB in memory".format(np.shape(
        data_attribute), data_attribute.size * data_attribute.itemsize / 1024 / 1024 / 1024))
    print("data_feature: {}, {}GB in memory".format(np.shape(data_feature),
          data_feature.size * data_feature.itemsize / 1024 / 1024 / 1024))
    print("data_gen_flag: {}, {}GB in memory".format(np.shape(
        data_gen_flag), data_gen_flag.size * data_gen_flag.itemsize / 1024 / 1024 / 1024))

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
