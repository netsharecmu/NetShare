import inspect
import pickle
import ipaddress
import copy
import time
import more_itertools
import os
import math
import json
import pickle
import ctypes
import pkgutil
import shutil

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import netshare.ray as ray

from gensim.models import Word2Vec
from tqdm import tqdm

from .word2vec_embedding import word2vec_train
from .preprocess_helper import countList2cdf, continuous_list_flag, plot_cdf
from .preprocess_helper import df2chunks, split_per_chunk
from ..pre_post_processor import PrePostProcessor
from netshare.utils import Tee, Output, output
from netshare.utils import Normalization
from netshare.utils import ContinuousField, DiscreteField, BitField
from netshare.utils import exec_cmd


class NetsharePrePostProcessor(PrePostProcessor):
    def _pre_process(self, input_folder, output_folder, log_folder):
        print(f"{self.__class__.__name__}.{inspect.stack()[0][3]}")

        # single file
        if os.path.isfile(input_folder):
            print(input_folder)
        # multiple files
        else:
            print("Merged file is located at {}")

        fields = {}

        # load data as csv
        if self._config["dataset_type"] == "pcap":
            if input_folder.endswith(".csv"):
                shutil.copyfile(
                    os.path.join(input_folder),
                    os.path.join(output_folder, "raw.csv")
                )
                df = pd.read_csv(input_folder)
            else:
                # compile shared library for converting pcap to csv
                cwd = os.path.dirname(os.path.abspath(__file__))
                cmd = f"cd {cwd} && \
                    cc -fPIC -shared -o pcap2csv.so main.c -lm -lpcap"
                exec_cmd(cmd, wait=True)

                pcap2csv_func = ctypes.CDLL(
                    os.path.join(cwd, "pcap2csv.so")).pcap2csv
                pcap2csv_func.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
                csv_file = os.path.join(
                    output_folder,
                    os.path.splitext(os.path.basename(input_folder))[0] +
                    ".csv")
                pcap2csv_func(
                    input_folder.encode('utf-8'),  # pcap file
                    csv_file.encode('utf-8')  # csv file
                )
                print(f"{input_folder} has been converted to {csv_file}")
                df = pd.read_csv(csv_file)
        elif self._config["dataset_type"] == "netflow":
            df = pd.read_csv(input_folder)
            df.to_csv(os.path.join(output_folder, "raw.csv"), index=False)
        else:
            raise ValueError("Only PCAP and NetFlow are currently supported!")

        # train/load word2vec model
        if self._config["encode_IP"] not in ['bit', 'word2vec']:
            raise ValueError("IP can be only encoded as `bit` or `word2vec`!")
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

        fields = {}

        if self._config["encode_IP"] == 'bit':
            fields["srcip"] = BitField(
                name="srcip",
                num_bits=32
            )

            fields["dstip"] = BitField(
                name="dstip",
                num_bits=32
            )

        for i in range(self._config["word2vec_vecSize"]):
            if self._config["encode_IP"] == 'word2vec':
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

        if self._config["timestamp"] == "interarrival":
            fields["flow_start"] = ContinuousField(
                name="flow_start",
                norm_option=NORM_OPTION
            )
            fields["interarrival_within_flow"] = ContinuousField(
                name="interarrival_within_flow",
                norm_option=NORM_OPTION
            )
        elif self._config["timestamp"] == "raw":
            if self._config["dataset_type"] == "pcap":
                fields["time"] = ContinuousField(
                    name="time",
                    norm_option=NORM_OPTION
                )
            elif self._config["dataset_type"] == "netflow":
                fields["ts"] = ContinuousField(
                    name="ts",
                    norm_option=NORM_OPTION
                )
        else:
            raise ValueError("Timestamp encoding can be only \
                `interarrival` or 'raw")

        if self._config["dataset_type"] == "pcap":
            fields["pkt_len"] = ContinuousField(
                name="pkt_len",
                min_x=(float(df["pkt_len"].min())
                       if not self._config["dp"]
                       else 20.0),
                max_x=(float(df["pkt_len"].max())
                       if not self._config["dp"]
                       else 1500.0),
                norm_option=NORM_OPTION
            )
            df["pkt_len"] = fields["pkt_len"].normalize(df["pkt_len"])

            if self._config["full_IP_header"]:
                fields["tos"] = ContinuousField(
                    name="tos",
                    min_x=0.0,
                    max_x=255.0,
                    norm_option=NORM_OPTION
                )
                fields["id"] = ContinuousField(
                    name="id",
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
                    min_x=0.0,
                    max_x=8191.0,  # 13 bits
                    norm_option=NORM_OPTION
                )
                fields["ttl"] = ContinuousField(
                    name="ttl",
                    # TTL less than 1 will be rounded up to 1;
                    # TTL=0 will be dropped.
                    min_x=1.0,
                    max_x=255.0,
                    norm_option=NORM_OPTION
                )
                for field in ["tos", "id", "off", "ttl"]:
                    df[field] = fields[field].normalize(df[field])

        elif self._config["dataset_type"] == "netflow":
            # log-transform
            df["td"] = np.log(1+df["td"])
            df["pkt"] = np.log(1+df["pkt"])
            df["byt"] = np.log(1+df["byt"])

            fields["td"] = ContinuousField(
                name="td",
                min_x=float(df["td"].min()),
                max_x=float(df["td"].max()),
                norm_option=NORM_OPTION
            )
            df["td"] = fields["td"].normalize(df["td"])

            fields["pkt"] = ContinuousField(
                name="pkt",
                min_x=float(df["pkt"].min()),
                max_x=float(df["pkt"].max()),
                norm_option=NORM_OPTION
            )
            df["pkt"] = fields["pkt"].normalize(df["pkt"])

            fields["byt"] = ContinuousField(
                name="byt",
                min_x=float(df["byt"].min()),
                max_x=float(df["byt"].max()),
                norm_option=NORM_OPTION
            )
            df["byt"] = fields["byt"].normalize(df["byt"])

            for field in ['label', 'type']:
                if field in df.columns:
                    fields[field] = DiscreteField(
                        name=field,
                        choices=list(set(df[field]))
                    )

        '''define metadata and measurement'''
        metadata = ["srcip", "dstip", "srcport", "dstport", "proto"]
        measurement = list(set(df.columns) - set(metadata))
        gk = df.groupby(by=metadata)

        # flow size distribution
        plot_cdf(
            count_list=sorted(gk.size().values, reverse=True),
            xlabel="# of packets per flow",
            ylabel="CDF",
            title="",
            filename="raw_pkts_per_flow.png",
            base_dir=output_folder
        )

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
                chunk_id+1, len(df_chunks), len(flowkeys)))
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
            float(num_flows_cross_chunk)/len(flowkeys_chunklen_list)*100))
        print("# of non-continuous flows:", num_non_continuous_flows)
        plot_cdf(
            count_list=flowkeys_chunklen_list,
            xlabel="# of chunks per flow",
            ylabel="CDF",
            title="",
            filename="raw_flow_numchunks.png",
            base_dir=output_folder
        )

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
        if not self._config["max_flow_len"]:
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

        if self._config["dataset_type"] == "pcap":
            shutil.copyfile(
                os.path.join(input_folder, "best_syn_dfs", "syn.pcap"),
                os.path.join(output_folder, "syn.pcap")
            )
        elif self._config["dataset_type"] == "netflow":
            shutil.copyfile(
                os.path.join(input_folder, "best_syn_dfs", "syn.csv"),
                os.path.join(output_folder, "syn.csv")
            )

        return True
